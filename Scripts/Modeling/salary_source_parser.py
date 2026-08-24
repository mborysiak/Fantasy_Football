"""Parse the copy-pasted ESPN auction salary export.

The export is a vertical stream of records.  Each record starts with ``$``,
followed by its salary, player label, optional injury status, team, position,
and projection fields.  Player-name length is not part of that contract.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


ESPN_POSITION_TOKENS = frozenset({"QB", "RB", "WR", "TE", "K", "D/ST"})
ESPN_TEAM_ALIAS_TOKENS = frozenset(
    {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "FA",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LAC",
        "LAR",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WSH",
    }
)
ESPN_INJURY_STATUS_TOKENS = frozenset(
    {
        "D",
        "IR",
        "NA",
        "NFI",
        "O",
        "P",
        "PUP",
        "Q",
        "SSPD",
        "SUSP",
    }
)


class SalarySourceFormatError(ValueError):
    """Raised when the ESPN salary stream cannot be parsed one-to-one."""


class SalaryPopulationContractError(ValueError):
    """Raised when a V2 salary fallback row lacks a model input."""


def collapse_identical_salary_rows(
    frame: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """Collapse literal repeats while rejecting conflicting player-year rows."""
    deduplicated = frame.drop_duplicates().reset_index(drop=True)
    duplicate_keys = deduplicated.duplicated(
        ["player", "year"], keep=False
    )
    if duplicate_keys.any():
        duplicates = (
            deduplicated.loc[duplicate_keys, ["player", "year"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise SalaryPopulationContractError(
            f"{source_name} has conflicting duplicate player-year rows "
            f"after name cleaning: {duplicates[:10]}"
        )
    return deduplicated


@dataclass(frozen=True)
class SalarySourceSpec:
    """One governed ESPN salary export consumed by a production cycle."""

    year: int
    league: str
    path: Path
    expected_count: int | None
    require_terminal_zero: bool = False

    @property
    def source_name(self) -> str:
        return f"{self.year} {self.league} ESPN salary export"


_GOVERNED_SALARY_SLICES = {
    2026: (
        (2025, "beta", 200, False),
        (2025, "nv", 160, False),
        # The live ESPN player pool changes during the preseason.  Govern the
        # copied boundary (through a terminal $0 record), not a mutable row
        # count; the stage manifest separately freezes the exact file hash.
        (2026, "beta", None, True),
        # The active NV export is copied to the same practical terminal-$0
        # boundary as beta. Its exact contents remain hash-bound by the stage
        # manifest even though ESPN can change the displayed pool depth.
        (2026, "nv", None, True),
    ),
}

# Current V2 team labels remain null when trusted sources tie.  These two 2026
# production players are the only governed null-team rows that require the
# salary-only fallback; team is not a salary-model feature.  A new null-team
# fallback must be reviewed and added explicitly rather than passing silently.
_GOVERNED_SALARY_FALLBACK_NULL_TEAM_KEYS = {
    2026: frozenset(
        {
            "922c769f-2f60-5de7-a75b-322112ec9540",  # Stefon Diggs
            "d6609ec3-d6b5-5aa7-b18a-1e31e52f29e7",  # Deebo Samuel Sr.
        }
    ),
}
SALARY_FALLBACK_TEAM_POLICY_VERSION = "v2_nullable_team_conflict_v1"
V2_SALARY_VALIDATION_METHOD = "conditional_ppg_primary_blend"
V2_SALARY_VALIDATION_TARGET = "conditional_ppg"
V2_SALARY_RESIDUAL_PERCENTILES = (5, 10, 25, 75, 90, 95)
V2_SALARY_UNCERTAINTY_SOURCE = (
    "v2_locked_oos_strict_prior_position_residual_quantiles"
)


def build_causal_v2_salary_validation_features(
    predictions: pd.DataFrame,
    identities: pd.DataFrame,
    *,
    current_year: int,
) -> pd.DataFrame:
    """Build salary features from locked V2 OOS point predictions.

    A target season's uncertainty is estimated only from locked residuals in
    earlier seasons at the same position.  The first locked season is omitted
    because it has no strictly prior residual pool.
    """
    prediction_columns = {
        "player_key",
        "season",
        "position",
        "target_name",
        "method",
        "prediction",
        "residual",
    }
    missing_predictions = prediction_columns.difference(predictions.columns)
    if missing_predictions:
        raise SalaryPopulationContractError(
            "Locked V2 salary validation predictions are missing columns: "
            f"{sorted(missing_predictions)}."
        )
    identity_columns = {"player_key", "display_name"}
    missing_identities = identity_columns.difference(identities.columns)
    if missing_identities:
        raise SalaryPopulationContractError(
            "V2 salary validation identities are missing columns: "
            f"{sorted(missing_identities)}."
        )

    locked = predictions.loc[
        predictions["target_name"].eq(V2_SALARY_VALIDATION_TARGET)
        & predictions["method"].eq(V2_SALARY_VALIDATION_METHOD)
    ].copy()
    locked["season"] = pd.to_numeric(locked["season"], errors="coerce")
    locked["prediction"] = pd.to_numeric(
        locked["prediction"], errors="coerce"
    )
    locked["residual"] = pd.to_numeric(locked["residual"], errors="coerce")
    locked["position"] = (
        locked["position"].astype("string").str.strip().str.upper()
    )
    locked = locked.loc[
        locked["season"].lt(int(current_year))
        & locked["position"].isin({"QB", "RB", "WR", "TE"})
    ].dropna(
        subset=["player_key", "season", "position", "prediction", "residual"]
    )
    locked["season"] = locked["season"].astype(int)
    duplicate_keys = locked.duplicated(
        ["player_key", "season", "position"], keep=False
    )
    if locked.empty or duplicate_keys.any():
        duplicate_values = locked.loc[
            duplicate_keys,
            ["player_key", "season", "position"],
        ].head(20).to_dict("records")
        raise SalaryPopulationContractError(
            "Locked V2 salary validation rows must be non-empty and unique by "
            "player-season-position; duplicates="
            f"{duplicate_values}."
        )

    identity_bridge = identities[["player_key", "display_name"]].drop_duplicates()
    if (
        identity_bridge["player_key"].isna().any()
        or identity_bridge["display_name"].isna().any()
        or identity_bridge["player_key"].duplicated().any()
    ):
        raise SalaryPopulationContractError(
            "V2 salary validation identities must provide one display name per "
            "player key."
        )
    locked = locked.merge(
        identity_bridge,
        on="player_key",
        how="left",
        validate="many_to_one",
    )
    if locked["display_name"].isna().any():
        missing_keys = locked.loc[
            locked["display_name"].isna(), "player_key"
        ].head(20).tolist()
        raise SalaryPopulationContractError(
            "Locked V2 salary validation rows lack canonical identities: "
            f"{missing_keys}."
        )

    records: list[pd.DataFrame] = []
    for target_season in sorted(locked["season"].unique()):
        target = locked.loc[locked["season"].eq(target_season)].copy()
        for position, position_target in target.groupby("position", sort=True):
            donors = locked.loc[
                locked["season"].lt(target_season)
                & locked["position"].eq(position),
                "residual",
            ]
            if donors.empty:
                # The earliest locked season has no causal uncertainty pool.
                continue
            quantiles = donors.quantile(
                [percentile / 100 for percentile in V2_SALARY_RESIDUAL_PERCENTILES]
            ).to_numpy()
            for percentile, value in zip(
                V2_SALARY_RESIDUAL_PERCENTILES,
                quantiles,
            ):
                position_target[f"pred_resid_{percentile}"] = float(value)
            records.append(position_target)

    if not records:
        raise SalaryPopulationContractError(
            "Locked V2 salary validation history has no seasons with strictly "
            "prior residual donors."
        )
    result = pd.concat(records, ignore_index=True)
    residual_columns = [
        f"pred_resid_{percentile}"
        for percentile in V2_SALARY_RESIDUAL_PERCENTILES
    ]
    if result[residual_columns].isna().any().any():
        raise SalaryPopulationContractError(
            "Locked V2 salary validation uncertainty quantiles are incomplete."
        )
    result["ensemble_uncertainty_feature_source"] = (
        V2_SALARY_UNCERTAINTY_SOURCE
    )
    return result.rename(
        columns={
            "display_name": "player",
            "season": "year",
            "position": "pos",
            "prediction": "pred_fp_per_game",
        }
    )[
        [
            "player",
            "year",
            "pos",
            "pred_fp_per_game",
            *residual_columns,
            "ensemble_uncertainty_feature_source",
        ]
    ]


def governed_salary_source_specs(
    data_root: Path,
    cycle_year: int,
) -> tuple[SalarySourceSpec, ...]:
    """Return the exact salary exports approved for a production refresh."""
    try:
        governed_slices = _GOVERNED_SALARY_SLICES[int(cycle_year)]
    except KeyError as exc:
        raise SalarySourceFormatError(
            f"Production cycle {cycle_year} has no governed salary-source contract."
        ) from exc

    salary_root = Path(data_root) / "OtherData" / "Salaries"
    return tuple(
        SalarySourceSpec(
            year=year,
            league=league,
            path=salary_root / f"salaries_{year}_{league}.csv",
            expected_count=expected_count,
            require_terminal_zero=require_terminal_zero,
        )
        for year, league, expected_count, require_terminal_zero in governed_slices
    )


def governed_salary_fallback_null_team_keys(
    cycle_year: int,
) -> frozenset[str]:
    """Return reviewed null-team salary fallback keys for a production cycle."""
    try:
        return _GOVERNED_SALARY_FALLBACK_NULL_TEAM_KEYS[int(cycle_year)]
    except KeyError as exc:
        raise SalaryPopulationContractError(
            f"Production cycle {cycle_year} has no governed null-team "
            "salary-fallback contract."
        ) from exc


def _is_position_token(token: str) -> bool:
    # ESPN can append a real-life secondary position (for example Travis
    # Hunter's ``WR, CB``) while the first token remains the fantasy position.
    primary_position = token.upper().split(",", maxsplit=1)[0].strip()
    return primary_position in ESPN_POSITION_TOKENS


def _nonblank_tokens(source: pd.DataFrame) -> list[tuple[int, str]]:
    if not isinstance(source, pd.DataFrame):
        raise TypeError("ESPN salary source must be a pandas DataFrame.")
    if source.shape[1] != 1:
        raise SalarySourceFormatError(
            "ESPN salary source must contain exactly one vertical value column."
        )

    tokens: list[tuple[int, str]] = []
    for row_number, value in enumerate(source.iloc[:, 0], start=1):
        if pd.isna(value):
            continue
        token = str(value).strip()
        if token:
            tokens.append((row_number, token))
    return tokens


def _parse_salary(token: str, *, record_number: int, row_number: int) -> int:
    normalized = token.replace(",", "")
    try:
        salary = float(normalized)
    except ValueError as exc:
        raise SalarySourceFormatError(
            f"Record {record_number} row {row_number} salary must be numeric; "
            f"found {token!r}."
        ) from exc

    if not math.isfinite(salary) or salary < 0 or not salary.is_integer():
        raise SalarySourceFormatError(
            f"Record {record_number} row {row_number} salary must be a "
            f"non-negative whole dollar amount; found {token!r}."
        )
    return int(salary)


def _parse_record(
    segment: list[tuple[int, str]],
    *,
    record_number: int,
    marker_row: int,
) -> dict[str, object]:
    if len(segment) < 4:
        raise SalarySourceFormatError(
            f"Record {record_number} beginning at row {marker_row} is incomplete; "
            "expected salary, player, team, and position fields."
        )

    salary_row, salary_token = segment[0]
    salary = _parse_salary(
        salary_token,
        record_number=record_number,
        row_number=salary_row,
    )
    player_row, player = segment[1]
    normalized_player = player.upper()
    if (
        player == "$"
        or _is_position_token(player)
        or normalized_player in ESPN_INJURY_STATUS_TOKENS
        or not any(character.isalpha() for character in player)
    ):
        raise SalarySourceFormatError(
            f"Record {record_number} row {player_row} has an invalid player "
            f"label: {player!r}."
        )

    position_indices = [
        index
        for index, (_, token) in enumerate(segment[2:], start=2)
        if _is_position_token(token)
    ]
    if not position_indices:
        raise SalarySourceFormatError(
            f"Record {record_number} beginning at row {marker_row} has no "
            "recognized ESPN position field."
        )
    if len(position_indices) > 1:
        raise SalarySourceFormatError(
            f"Record {record_number} beginning at row {marker_row} contains "
            "multiple position fields; a '$' record marker may be missing."
        )
    if position_indices[0] < 3:
        raise SalarySourceFormatError(
            f"Record {record_number} beginning at row {marker_row} is missing "
            "a player or team field before its position."
        )

    position_index = position_indices[0]
    team_row, team = segment[position_index - 1]
    if team.upper() not in ESPN_TEAM_ALIAS_TOKENS:
        raise SalarySourceFormatError(
            f"Record {record_number} row {team_row} expected a governed ESPN "
            f"team alias immediately before its position; found {team!r}."
        )
    if position_index not in (3, 4):
        raise SalarySourceFormatError(
            f"Record {record_number} beginning at row {marker_row} has an "
            "unexpected layout between player and team."
        )
    if position_index == 4:
        status_row, status = segment[2]
        if status.upper() not in ESPN_INJURY_STATUS_TOKENS:
            raise SalarySourceFormatError(
                f"Record {record_number} row {status_row} expected a governed "
                f"ESPN injury status; found {status!r}."
            )

    return {"player": player, "salary": salary}


def validate_salary_records(
    records: pd.DataFrame,
    *,
    source_name: str = "ESPN salary records",
) -> None:
    """Fail closed unless parsed player/salary rows are one-to-one and valid."""
    required = {"player", "salary"}
    missing = required - set(records.columns)
    if missing:
        raise SalarySourceFormatError(
            f"{source_name} is missing required columns: {sorted(missing)}."
        )
    if records.empty:
        raise SalarySourceFormatError(f"{source_name} contains no salary records.")
    if records[["player", "salary"]].isna().any().any():
        raise SalarySourceFormatError(
            f"{source_name} contains a missing player or salary."
        )

    normalized_players = records.player.astype(str).str.strip().str.casefold()
    empty_players = normalized_players.eq("")
    if empty_players.any():
        raise SalarySourceFormatError(f"{source_name} contains a blank player label.")
    duplicates = normalized_players.duplicated(keep=False)
    if duplicates.any():
        duplicate_names = sorted(
            records.loc[duplicates, "player"].astype(str).drop_duplicates().tolist()
        )
        raise SalarySourceFormatError(
            f"{source_name} contains duplicate player records: "
            f"{duplicate_names[:10]}."
        )

    salary = pd.to_numeric(records.salary, errors="coerce")
    invalid_salary = (
        salary.isna()
        | ~salary.map(math.isfinite)
        | salary.lt(0)
        | salary.mod(1).ne(0)
    )
    if invalid_salary.any():
        bad_values = records.loc[invalid_salary, "salary"].head(10).tolist()
        raise SalarySourceFormatError(
            f"{source_name} contains invalid whole-dollar salaries: {bad_values}."
        )


def validate_v2_salary_fallback_context(
    fallback: pd.DataFrame,
    *,
    allowed_unresolved_team_player_keys: Iterable[str] = (),
) -> dict[str, object]:
    """Validate salary inputs while permitting governed unresolved teams.

    V2 deliberately leaves ``team`` null when trusted team labels tie.  The
    salary model does not consume team (its modeling frames replace the field
    with a placeholder), so team remains required as an auditable column but
    may be null only for an explicitly allowed row with ``team_conflict=1``.
    Every actual salary-model input remains fail-closed.
    """
    required_model_context = (
        "position",
        "year_exp",
        "adp_median",
        "adp_log",
        "expert_points_median",
    )
    required_columns = {
        "player_key",
        "player",
        "team",
        "team_conflict",
        *required_model_context,
    }
    missing_columns = required_columns.difference(fallback.columns)
    if missing_columns:
        raise SalaryPopulationContractError(
            "V2 salary population fallback is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    duplicate_keys = fallback["player_key"].duplicated(keep=False)
    if fallback["player_key"].isna().any() or duplicate_keys.any():
        bad_keys = fallback.loc[
            fallback["player_key"].isna() | duplicate_keys,
            "player_key",
        ].head(20).tolist()
        raise SalaryPopulationContractError(
            "V2 salary population fallback lacks unique player keys: "
            f"{bad_keys}."
        )

    missing_context = fallback[list(required_model_context)].isna().any(axis=1)
    if missing_context.any():
        missing = fallback.loc[
            missing_context,
            ["player_key", "player", *required_model_context],
        ].to_dict("records")
        raise SalaryPopulationContractError(
            "V2 salary population fallback lacks required model context: "
            f"{missing}."
        )

    normalized_position = (
        fallback["position"].astype("string").str.strip().str.upper()
    )
    invalid_position = ~normalized_position.isin({"QB", "RB", "WR", "TE"})
    if invalid_position.any():
        invalid = fallback.loc[
            invalid_position,
            ["player_key", "player", "position"],
        ].to_dict("records")
        raise SalaryPopulationContractError(
            "V2 salary population fallback contains an invalid fantasy "
            f"position: {invalid}."
        )

    team_conflict = pd.to_numeric(
        fallback["team_conflict"], errors="coerce"
    )
    invalid_team_conflict = team_conflict.isna() | ~team_conflict.isin({0, 1})
    if invalid_team_conflict.any():
        invalid = fallback.loc[
            invalid_team_conflict,
            ["player_key", "player", "team_conflict"],
        ].to_dict("records")
        raise SalaryPopulationContractError(
            "V2 salary population fallback contains an invalid team_conflict "
            f"audit value: {invalid}."
        )

    normalized_team = fallback["team"].astype("string").str.strip()
    unresolved_team = normalized_team.isna() | normalized_team.eq("")
    unresolved_without_conflict = unresolved_team & ~team_conflict.eq(1)
    if unresolved_without_conflict.any():
        invalid = fallback.loc[
            unresolved_without_conflict,
            ["player_key", "player", "team", "team_conflict"],
        ].to_dict("records")
        raise SalaryPopulationContractError(
            "V2 salary population fallback has unresolved teams without a "
            f"governed source conflict: {invalid}."
        )
    unresolved_team_player_keys = set(
        fallback.loc[unresolved_team, "player_key"].astype(str)
    )
    allowed_unresolved_team_player_keys = {
        str(player_key)
        for player_key in allowed_unresolved_team_player_keys
    }
    ungoverned_team_keys = (
        unresolved_team_player_keys - allowed_unresolved_team_player_keys
    )
    if ungoverned_team_keys:
        raise SalaryPopulationContractError(
            "V2 salary population fallback contains ungoverned unresolved "
            f"team keys: {sorted(ungoverned_team_keys)}."
        )
    return {
        "policy_version": SALARY_FALLBACK_TEAM_POLICY_VERSION,
        "row_count": int(len(fallback)),
        "unresolved_team_count": int(unresolved_team.sum()),
        "unresolved_team_player_keys": sorted(unresolved_team_player_keys),
        "allowed_unresolved_team_player_keys": sorted(
            allowed_unresolved_team_player_keys
        ),
    }


def parse_espn_salary_records(
    source: pd.DataFrame,
    *,
    expected_count: int | None = None,
    source_name: str = "ESPN salary source",
) -> pd.DataFrame:
    """Return one validated player/salary row for every ``$`` record marker."""
    tokens = _nonblank_tokens(source)
    marker_indices = [index for index, (_, token) in enumerate(tokens) if token == "$"]
    if not marker_indices:
        raise SalarySourceFormatError(
            "ESPN salary source contains no '$' record markers."
        )
    if expected_count is not None and len(marker_indices) != int(expected_count):
        raise SalarySourceFormatError(
            f"{source_name} expected {int(expected_count)} '$' record markers; "
            f"found {len(marker_indices)}."
        )
    if marker_indices[0] != 0:
        prefix_row, prefix_value = tokens[0]
        raise SalarySourceFormatError(
            f"ESPN salary source has unexpected content before its first '$' "
            f"marker at row {prefix_row}: {prefix_value!r}."
        )

    parsed: list[dict[str, object]] = []
    errors: list[str] = []
    boundaries = marker_indices[1:] + [len(tokens)]
    for record_number, (marker_index, next_marker_index) in enumerate(
        zip(marker_indices, boundaries),
        start=1,
    ):
        marker_row = tokens[marker_index][0]
        segment = tokens[marker_index + 1 : next_marker_index]
        try:
            parsed.append(
                _parse_record(
                    segment,
                    record_number=record_number,
                    marker_row=marker_row,
                )
            )
        except SalarySourceFormatError as exc:
            errors.append(str(exc))

    if errors or len(parsed) != len(marker_indices):
        detail = "; ".join(errors[:5])
        raise SalarySourceFormatError(
            f"Parsed {len(parsed)} complete records from "
            f"{len(marker_indices)} salary markers. {detail}"
        )

    records = pd.DataFrame(parsed, columns=["player", "salary"])
    if expected_count is not None and len(records) != int(expected_count):
        raise SalarySourceFormatError(
            f"{source_name} expected {int(expected_count)} parsed records; "
            f"found {len(records)}."
        )
    validate_salary_records(records, source_name=source_name)
    return records


def _prepare_salary_source_slice(
    spec: SalarySourceSpec,
    *,
    name_clean: Callable[[str], str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not spec.path.is_file():
        raise FileNotFoundError(
            f"Missing governed salary source for {spec.year} {spec.league}: "
            f"{spec.path}"
        )
    try:
        source = pd.read_csv(spec.path, header=None)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise SalarySourceFormatError(
            f"Could not read {spec.source_name} at {spec.path}: {exc}"
        ) from exc

    marker_count = sum(
        token == "$"
        for _, token in _nonblank_tokens(source)
    )
    records = parse_espn_salary_records(
        source,
        expected_count=spec.expected_count,
        source_name=spec.source_name,
    )
    terminal_salary = int(records.iloc[-1].salary)
    if spec.require_terminal_zero and terminal_salary != 0:
        raise SalarySourceFormatError(
            f"{spec.source_name} must end at an ESPN $0 salary record; "
            f"found terminal salary ${terminal_salary}."
        )
    records["player"] = records.player.map(
        lambda player: name_clean(str(player))
    )
    validate_salary_records(
        records,
        source_name=f"name-cleaned {spec.source_name}",
    )
    records["year"] = int(spec.year)
    records["league"] = str(spec.league)
    return records, {
        "year": int(spec.year),
        "league": str(spec.league),
        "source": str(spec.path.resolve()),
        "expected_count": (
            None if spec.expected_count is None else int(spec.expected_count)
        ),
        "require_terminal_zero": bool(spec.require_terminal_zero),
        "terminal_salary": terminal_salary,
        "marker_count": int(marker_count),
        "parsed_count": int(len(records)),
    }


def _salary_schema_state(
    connection: sqlite3.Connection,
) -> tuple[str, tuple[tuple[str, str | None], ...]]:
    table_row = connection.execute(
        """SELECT sql
             FROM sqlite_master
            WHERE type='table' AND name='Salaries'"""
    ).fetchone()
    if table_row is None:
        raise SalarySourceFormatError(
            "Staged Simulation database has no Salaries table."
        )
    columns = {
        row[1]
        for row in connection.execute('PRAGMA table_info("Salaries")')
    }
    required = {"player", "salary", "year", "league"}
    missing = required.difference(columns)
    if missing:
        raise SalarySourceFormatError(
            f"Staged Salaries table is missing columns: {sorted(missing)}."
        )
    indexes = tuple(
        connection.execute(
            """SELECT name, sql
                 FROM sqlite_master
                WHERE type='index' AND tbl_name='Salaries'
             ORDER BY name"""
        ).fetchall()
    )
    return str(table_row[0]), indexes


def repair_governed_salary_slices(
    database_path: Path,
    specs: Iterable[SalarySourceSpec],
    *,
    name_clean: Callable[[str], str],
    live_database_path: Path,
) -> dict[str, object]:
    """Atomically restore governed salary slices in a staged database only."""
    database_path = Path(database_path).resolve()
    live_database_path = Path(live_database_path).resolve()
    if database_path == live_database_path:
        raise PermissionError(
            "Governed salary repair refuses to modify the live Simulation database."
        )
    if not database_path.is_file():
        raise FileNotFoundError(
            f"Staged Simulation database does not exist: {database_path}"
        )

    specs = tuple(specs)
    slices = [(int(spec.year), str(spec.league)) for spec in specs]
    if not specs:
        raise SalarySourceFormatError("No governed salary slices were provided.")
    if len(set(slices)) != len(slices):
        raise SalarySourceFormatError(
            f"Governed salary contract repeats a year/league slice: {slices}."
        )

    # Parse and validate every source before opening the write transaction.
    prepared = [
        _prepare_salary_source_slice(spec, name_clean=name_clean)
        for spec in specs
    ]

    connection = sqlite3.connect(database_path)
    try:
        schema_before = _salary_schema_state(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            for records, receipt in prepared:
                year = int(receipt["year"])
                league = str(receipt["league"])
                connection.execute(
                    "DELETE FROM Salaries WHERE year=? AND league=?",
                    (year, league),
                )
                connection.executemany(
                    """INSERT INTO Salaries (player, salary, year, league)
                       VALUES (?, ?, ?, ?)""",
                    records[["player", "salary", "year", "league"]]
                    .itertuples(index=False, name=None),
                )

            for _, receipt in prepared:
                count, unique_count = connection.execute(
                    """SELECT COUNT(*),
                              COUNT(DISTINCT player COLLATE NOCASE)
                         FROM Salaries
                        WHERE year=? AND league=?""",
                    (int(receipt["year"]), str(receipt["league"])),
                ).fetchone()
                source_count = int(receipt["parsed_count"])
                if count != source_count or unique_count != source_count:
                    raise SalarySourceFormatError(
                        f"Post-write Salaries validation failed for "
                        f"{receipt['year']} {receipt['league']}: expected the "
                        f"parsed source's {source_count} rows and unique players; found "
                        f"{count} rows and {unique_count} unique players."
                    )

            if _salary_schema_state(connection) != schema_before:
                raise SalarySourceFormatError(
                    "Governed salary repair changed the Salaries schema or indexes."
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    finally:
        connection.close()

    return {
        "database": str(database_path),
        "slices": [receipt for _, receipt in prepared],
    }
