import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from Scripts.Modeling.salary_source_parser import (
    SalaryPopulationContractError,
    SalarySourceSpec,
    SalarySourceFormatError,
    governed_salary_fallback_null_team_keys,
    governed_salary_source_specs,
    parse_espn_salary_records,
    repair_governed_salary_slices,
    validate_v2_salary_fallback_context,
)


def _source(*values):
    return pd.DataFrame({0: values})


def _write_salary_source(path, *records):
    values = []
    for player, salary in records:
        values.extend(("$", str(salary), player, "DEN", "RB", "0-0"))
    _source(*values).to_csv(path, index=False, header=False)


def _salary_database(path, *, salary_constraint=""):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            f"""CREATE TABLE Salaries (
                    player TEXT,
                    salary REAL {salary_constraint},
                    year INTEGER,
                    league TEXT,
                    std_dev REAL,
                    min_score REAL,
                    max_score REAL
                )"""
        )
        connection.execute(
            """CREATE INDEX ix_salary_slice
               ON Salaries(year, league, player)"""
        )
        connection.executemany(
            "INSERT INTO Salaries(player, salary, year, league) VALUES (?, ?, ?, ?)",
            (
                ("Old Beta", 1, 2025, "beta"),
                ("Old NV", 1, 2025, "nv"),
                ("Old Current", 1, 2026, "beta"),
                ("Unrelated", 1, 2024, "beta"),
            ),
        )
        connection.commit()


def _schema_state(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(
            """SELECT type, name, sql
                 FROM sqlite_master
                WHERE (type='table' AND name='Salaries')
                   OR (type='index' AND tbl_name='Salaries')
             ORDER BY type, name"""
        ).fetchall()


def test_2026_governed_salary_contract_is_exact():
    specs = governed_salary_source_specs(Path("Data"), 2026)

    assert [
        (
            spec.year,
            spec.league,
            spec.expected_count,
            spec.require_terminal_zero,
            spec.path.name,
        )
        for spec in specs
    ] == [
        (2025, "beta", 200, False, "salaries_2025_beta.csv"),
        (2025, "nv", 160, False, "salaries_2025_nv.csv"),
        (2026, "beta", None, True, "salaries_2026_beta.csv"),
    ]


def test_structural_parser_keeps_bo_nix_and_preserves_salary_alignment():
    source = _source(
        "$",
        "5",
        "Bo Nix",
        "DEN",
        "QB",
        "0-0",
        "$",
        "2",
        "Isaiah Likely",
        "Q",
        "BAL",
        "TE",
        "0-0",
    )

    parsed = parse_espn_salary_records(source)

    assert parsed.to_dict("records") == [
        {"player": "Bo Nix", "salary": 5},
        {"player": "Isaiah Likely", "salary": 2},
    ]


def test_structural_parser_accepts_espn_secondary_position_label():
    parsed = parse_espn_salary_records(
        _source("$", "12", "Travis Hunter", "Q", "JAX", "WR, CB", "0-0")
    )

    assert parsed.to_dict("records") == [
        {"player": "Travis Hunter", "salary": 12}
    ]


def test_structural_parser_accepts_optional_injury_status_and_free_agent_alias():
    parsed = parse_espn_salary_records(
        _source(
            "$",
            "5",
            "Bo Nix",
            "Q",
            "DEN",
            "QB",
            "0-0",
            "$",
            "2",
            "Free Agent",
            "FA",
            "RB",
            "0-0",
        )
    )

    assert parsed.to_dict("records") == [
        {"player": "Bo Nix", "salary": 5},
        {"player": "Free Agent", "salary": 2},
    ]


def _v2_salary_fallback_row(**overrides):
    row = {
        "player_key": "player-key",
        "player": "Current Player",
        "position": "WR",
        "team": "DEN",
        "team_conflict": 0,
        "year_exp": 3.0,
        "adp_median": 100.0,
        "adp_log": 4.6,
        "expert_points_median": 150.0,
        "consensus_room_share": 0.25,
    }
    row.update(overrides)
    return row


def test_v2_salary_fallback_accepts_governed_null_team_and_room_share():
    fallback = pd.DataFrame(
        [
            _v2_salary_fallback_row(
                team=pd.NA,
                team_conflict=1,
                consensus_room_share=pd.NA,
            )
        ]
    )

    receipt = validate_v2_salary_fallback_context(
        fallback,
        allowed_unresolved_team_player_keys={"player-key"},
    )

    assert receipt == {
        "policy_version": "v2_nullable_team_conflict_v1",
        "row_count": 1,
        "unresolved_team_count": 1,
        "unresolved_team_player_keys": ["player-key"],
        "allowed_unresolved_team_player_keys": ["player-key"],
    }


@pytest.mark.parametrize(
    "column",
    [
        "position",
        "year_exp",
        "adp_median",
        "adp_log",
        "expert_points_median",
    ],
)
def test_v2_salary_fallback_rejects_missing_model_context(column):
    fallback = pd.DataFrame([_v2_salary_fallback_row(**{column: pd.NA})])

    with pytest.raises(
        SalaryPopulationContractError,
        match="lacks required model context",
    ):
        validate_v2_salary_fallback_context(fallback)


def test_v2_salary_fallback_rejects_ungoverned_null_team():
    fallback = pd.DataFrame(
        [_v2_salary_fallback_row(team=pd.NA, team_conflict=1)]
    )

    with pytest.raises(
        SalaryPopulationContractError,
        match="ungoverned unresolved team keys",
    ):
        validate_v2_salary_fallback_context(fallback)


@pytest.mark.parametrize("team", [pd.NA, "   "])
def test_v2_salary_fallback_rejects_unresolved_team_without_conflict(team):
    fallback = pd.DataFrame(
        [_v2_salary_fallback_row(team=team, team_conflict=0)]
    )

    with pytest.raises(
        SalaryPopulationContractError,
        match="unresolved teams without a governed source conflict",
    ):
        validate_v2_salary_fallback_context(
            fallback,
            allowed_unresolved_team_player_keys={"player-key"},
        )


def test_v2_salary_fallback_accepts_zero_unresolved_teams():
    receipt = validate_v2_salary_fallback_context(
        pd.DataFrame([_v2_salary_fallback_row()])
    )

    assert receipt["unresolved_team_count"] == 0
    assert receipt["unresolved_team_player_keys"] == []


def test_v2_salary_fallback_requires_team_audit_column():
    fallback = pd.DataFrame([_v2_salary_fallback_row()]).drop(columns="team")

    with pytest.raises(
        SalaryPopulationContractError,
        match="missing required columns.*team",
    ):
        validate_v2_salary_fallback_context(fallback)


@pytest.mark.parametrize("position", ["K", "", "   "])
def test_v2_salary_fallback_rejects_invalid_fantasy_position(position):
    fallback = pd.DataFrame(
        [_v2_salary_fallback_row(position=position)]
    )

    with pytest.raises(
        SalaryPopulationContractError,
        match="invalid fantasy position",
    ):
        validate_v2_salary_fallback_context(fallback)


@pytest.mark.parametrize("team_conflict", [pd.NA, "bad", -1, 0.5, 2])
def test_v2_salary_fallback_rejects_invalid_team_conflict(team_conflict):
    fallback = pd.DataFrame(
        [_v2_salary_fallback_row(team_conflict=team_conflict)]
    )

    with pytest.raises(
        SalaryPopulationContractError,
        match="invalid team_conflict audit value",
    ):
        validate_v2_salary_fallback_context(fallback)


def test_2026_null_team_salary_fallback_contract_is_exact():
    assert governed_salary_fallback_null_team_keys(2026) == {
        "922c769f-2f60-5de7-a75b-322112ec9540",
        "d6609ec3-d6b5-5aa7-b18a-1e31e52f29e7",
    }


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            ("$", "5", "Bo Nix", "Q", "QB", "0-0"),
            "expected a governed ESPN team alias",
        ),
        (
            ("$", "5", "Q", "DEN", "QB", "0-0"),
            "invalid player label",
        ),
        (
            ("$", "5", "Bo Nix", "UNKNOWN", "DEN", "QB", "0-0"),
            "expected a governed ESPN injury status",
        ),
    ],
)
def test_structural_parser_rejects_shifted_team_or_status_tokens(values, message):
    with pytest.raises(SalarySourceFormatError, match=message):
        parse_espn_salary_records(_source(*values))


def test_structural_parser_rejects_marker_to_record_count_mismatch():
    source = _source(
        "$",
        "5",
        "Bo Nix",
        "DEN",
        "QB",
        "$",
        "2",
    )

    with pytest.raises(
        SalarySourceFormatError,
        match=r"Parsed 1 complete records from 2 salary markers",
    ):
        parse_espn_salary_records(source)


def test_structural_parser_enforces_governed_marker_count():
    source = _source(
        "$",
        "5",
        "Bo Nix",
        "DEN",
        "QB",
        "$",
        "2",
        "Isaiah Likely",
        "BAL",
        "TE",
    )

    with pytest.raises(
        SalarySourceFormatError,
        match=r"expected 3 '\$' record markers; found 2",
    ):
        parse_espn_salary_records(
            source,
            expected_count=3,
            source_name="governed test export",
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (("$", "five", "Bo Nix", "DEN", "QB"), "salary must be numeric"),
        (("$", "5", "DEN", "QB", "0-0"), "missing a player or team"),
        (("$", "5", "Bo Nix", "DEN", "0-0"), "no recognized ESPN position"),
    ],
)
def test_structural_parser_rejects_malformed_records(values, message):
    with pytest.raises(SalarySourceFormatError, match=message):
        parse_espn_salary_records(_source(*values))


def test_structural_parser_rejects_duplicate_player_records():
    source = _source(
        "$",
        "5",
        "Bo Nix",
        "DEN",
        "QB",
        "$",
        "4",
        "bo nix",
        "DEN",
        "QB",
    )

    with pytest.raises(SalarySourceFormatError, match="duplicate player records"):
        parse_espn_salary_records(source)


def test_governed_repair_is_atomic_and_preserves_schema_and_indexes(tmp_path):
    database = tmp_path / "staged.sqlite3"
    live_database = tmp_path / "live.sqlite3"
    _salary_database(database)
    schema_before = _schema_state(database)

    source_specs = []
    for year, league, player, salary in (
        (2025, "beta", "Beta.Player", 5),
        (2025, "nv", "NV Player", 6),
        (2026, "beta", "Current Player", 0),
    ):
        source = tmp_path / f"salaries_{year}_{league}.csv"
        _write_salary_source(source, (player, salary))
        source_specs.append(
            SalarySourceSpec(
                year,
                league,
                source,
                expected_count=None if year == 2026 else 1,
                require_terminal_zero=year == 2026,
            )
        )

    receipt = repair_governed_salary_slices(
        database,
        source_specs,
        name_clean=lambda player: player.replace(".", ""),
        live_database_path=live_database,
    )

    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            """SELECT player, salary, year, league
                 FROM Salaries
             ORDER BY year, league"""
        ).fetchall()
    assert rows == [
        ("Unrelated", 1.0, 2024, "beta"),
        ("BetaPlayer", 5.0, 2025, "beta"),
        ("NV Player", 6.0, 2025, "nv"),
        ("Current Player", 0.0, 2026, "beta"),
    ]
    assert _schema_state(database) == schema_before
    assert [
        (item["year"], item["league"], item["marker_count"], item["parsed_count"])
        for item in receipt["slices"]
    ] == [
        (2025, "beta", 1, 1),
        (2025, "nv", 1, 1),
        (2026, "beta", 1, 1),
    ]


def test_variable_length_governed_repair_requires_terminal_zero(tmp_path):
    database = tmp_path / "staged.sqlite3"
    live_database = tmp_path / "live.sqlite3"
    source = tmp_path / "salaries_2026_beta.csv"
    _salary_database(database)
    _write_salary_source(source, ("Current Player", 1))

    with pytest.raises(
        SalarySourceFormatError,
        match=r"must end at an ESPN \$0 salary record",
    ):
        repair_governed_salary_slices(
            database,
            (
                SalarySourceSpec(
                    2026,
                    "beta",
                    source,
                    expected_count=None,
                    require_terminal_zero=True,
                ),
            ),
            name_clean=lambda player: player,
            live_database_path=live_database,
        )


def test_governed_repair_rolls_back_every_slice_on_insert_failure(tmp_path):
    database = tmp_path / "staged.sqlite3"
    _salary_database(database, salary_constraint="CHECK(salary < 10)")
    before_schema = _schema_state(database)

    source_specs = []
    for year, league, player, salary in (
        (2025, "beta", "New Beta", 5),
        (2025, "nv", "New NV", 6),
        (2026, "beta", "Rejected Current", 12),
    ):
        source = tmp_path / f"salaries_{year}_{league}.csv"
        _write_salary_source(source, (player, salary))
        source_specs.append(
            SalarySourceSpec(year, league, source, expected_count=1)
        )

    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        repair_governed_salary_slices(
            database,
            source_specs,
            name_clean=lambda player: player,
            live_database_path=tmp_path / "live.sqlite3",
        )

    with closing(sqlite3.connect(database)) as connection:
        governed_rows = connection.execute(
            """SELECT player, year, league
                 FROM Salaries
                WHERE (year=2025 AND league IN ('beta', 'nv'))
                   OR (year=2026 AND league='beta')
             ORDER BY year, league"""
        ).fetchall()
    assert governed_rows == [
        ("Old Beta", 2025, "beta"),
        ("Old NV", 2025, "nv"),
        ("Old Current", 2026, "beta"),
    ]
    assert _schema_state(database) == before_schema


def test_governed_repair_rejects_name_clean_collision_before_write(tmp_path):
    database = tmp_path / "staged.sqlite3"
    _salary_database(database)
    source = tmp_path / "salaries_2025_beta.csv"
    _write_salary_source(source, ("A-B", 5), ("A B", 6))

    with pytest.raises(
        SalarySourceFormatError,
        match="name-cleaned .* duplicate player records",
    ):
        repair_governed_salary_slices(
            database,
            (SalarySourceSpec(2025, "beta", source, expected_count=2),),
            name_clean=lambda player: player.replace("-", " "),
            live_database_path=tmp_path / "live.sqlite3",
        )

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute(
            """SELECT player FROM Salaries
                WHERE year=2025 AND league='beta'"""
        ).fetchall() == [("Old Beta",)]


def test_governed_repair_refuses_live_database_path(tmp_path):
    database = tmp_path / "Simulation.sqlite3"
    _salary_database(database)

    with pytest.raises(PermissionError, match="refuses to modify the live"):
        repair_governed_salary_slices(
            database,
            (),
            name_clean=lambda player: player,
            live_database_path=database,
        )
