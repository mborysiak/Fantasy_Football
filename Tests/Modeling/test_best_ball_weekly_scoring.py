import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from Scripts.Modeling import s4_Best_Ball_Weekly as weekly_builder
import config as scoring_config


def _quarterback_week():
    return pd.DataFrame(
        [
            {
                "player": "Scoring Quarterback",
                "team": "TST",
                "season": 2025,
                "week": 1,
                "pass_yards_gained_sum": 400,
                "pass_pass_touchdown_sum": 1,
                "pass_interception_sum": 1,
                "sack_sum": 2,
                "pass_qb_dropback_sum": 20,
                "rush_rush_attempt_sum": 1,
                "rush_yards_gained_sum": 200,
                "rush_rush_touchdown_sum": 1,
                "rush_fumble_lost_sum": 1,
            }
        ]
    )


def _receiver_week():
    return pd.DataFrame(
        [
            {
                "player": "Scoring Receiver",
                "team": "TST",
                "season": 2025,
                "week": 1,
                "rec_complete_pass_sum": 10,
                "rec_yards_gained_sum": 200,
                "rec_pass_touchdown_sum": 1,
            }
        ]
    )


def _projection_context():
    columns = [
        "player",
        "player_key",
        "player_key_match_method",
        "pos",
        "team",
        "season",
        "avg_proj_points",
        "preseason_proj_ppg",
        "validation_pred_fp_per_game",
        "historical_pred_fp_per_game",
        "historical_projection_source",
        "legacy_historical_pred_fp_per_game",
        "legacy_historical_projection_source",
        "v2_historical_pred_fp_per_game",
        "v2_point_center_source",
        "v2_template_center_available",
        weekly_builder.V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN,
        weekly_builder.V2_TEMPLATE_CENTER_POSITION_COLUMN,
        weekly_builder.V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN,
        weekly_builder.V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN,
        "historical_center_policy",
        "v2_recenter_promoted",
        "validation_ensemble_sources",
        "avg_pick",
        "year_exp",
        "source_year_exp",
        "year_exp_source",
        "year_exp_uncapped_delta",
        "year_exp_bucket",
        "exp_bucket",
        "qb_team_rank",
        "qb_team_rank_bucket",
        "projection_rank_pct",
        "projection_decile",
        "projection_tier",
    ]
    row = dict.fromkeys(columns, 0)
    row.update(
        {
            "player": "Scoring Receiver",
            "player_key": "player-1",
            "player_key_match_method": "fixture",
            "pos": "WR",
            "team": "TST",
            "season": 2025,
            "avg_proj_points": 128.0,
            "preseason_proj_ppg": 8.0,
            "historical_pred_fp_per_game": 8.0,
            "historical_projection_source": "fixture",
            "legacy_historical_pred_fp_per_game": 8.0,
            "legacy_historical_projection_source": "fixture",
            "historical_center_policy": "fixture",
            "validation_ensemble_sources": "fixture",
            "avg_pick": 100.0,
            "year_exp_source": "fixture",
            "exp_bucket": "young",
            "qb_team_rank_bucket": "non_qb",
            "projection_decile": 5,
            "projection_tier": "middle",
        }
    )
    return pd.DataFrame([row])


def _historical_center_projection(
    *,
    season=2018,
    pos="QB",
    player="Fallback Quarterback",
    player_key="player-1",
):
    return pd.DataFrame(
        [
            {
                "player": player,
                "player_key": player_key,
                "pos": pos,
                "season": season,
                "historical_pred_fp_per_game": 18.5,
                "historical_projection_source": "legacy_fixture",
            }
        ]
    )


def _write_center_database(
    path,
    *,
    center=None,
    center_available=0,
    include_center_row=True,
    include_quarantine_proof=True,
    player_key="player-1",
    center_season=2018,
    center_position="QB",
):
    model_run_id = "locked-beta-run"
    feature_run_id = "milestone-3-beta"
    foundation_run_id = "milestone-2-beta"
    quarantine_rule = next(
        rule
        for rule in weekly_builder.SOURCE_ROW_EXCLUSIONS
        if rule["exclusion_id"]
        == weekly_builder.BETA_2018_QB_CENTER_FALLBACK_EXCLUSION_ID
    )
    policy = weekly_builder.source_row_exclusion_policy_receipt(
        foundation_run_id
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE locked_template_handoff "
            "(model_run_id TEXT, player_key TEXT, season INTEGER, "
            "position TEXT, historical_pred_fp_per_game REAL, "
            "point_center_source TEXT, "
            "template_center_available INTEGER)"
        )
        if include_center_row:
            connection.execute(
                "INSERT INTO locked_template_handoff "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    model_run_id,
                    player_key,
                    center_season,
                    center_position,
                    center,
                    "fixture_locked_model",
                    center_available,
                ),
            )
        connection.execute(
            "CREATE TABLE locked_candidate_runs "
            "(model_run_id TEXT, feature_run_id TEXT)"
        )
        connection.execute(
            "INSERT INTO locked_candidate_runs VALUES (?, ?)",
            (model_run_id, feature_run_id),
        )
        connection.execute(
            "CREATE TABLE build_runs "
            "(run_id TEXT, component TEXT, league TEXT, "
            "foundation_run_id TEXT, status TEXT)"
        )
        connection.execute(
            "INSERT INTO build_runs VALUES (?, 'milestone_3', 'beta', ?, "
            "'complete')",
            (feature_run_id, foundation_run_id),
        )
        connection.execute(
            "CREATE TABLE source_manifest "
            "(run_id TEXT, component TEXT, source_name TEXT, source_uri TEXT, "
            "source_sha256 TEXT, row_count INTEGER)"
        )
        connection.execute(
            "INSERT INTO source_manifest VALUES (?, ?, ?, ?, ?, ?)",
            (
                policy["run_id"],
                policy["component"],
                policy["source_name"],
                policy["source_uri"],
                policy["source_sha256"],
                policy["row_count"],
            ),
        )
        if include_quarantine_proof:
            connection.execute(
                "INSERT INTO source_manifest VALUES (?, ?, ?, ?, ?, ?)",
                (
                    feature_run_id,
                    "source_quarantine",
                    quarantine_rule["exclusion_id"],
                    quarantine_rule["reference"],
                    None,
                    50,
                ),
            )


def test_qb_scoring_is_explicitly_league_specific():
    dk = weekly_builder.add_fantasy_points(
        _quarterback_week(),
        "QB",
        league="dk",
        filter_qb_workload=False,
    ).iloc[0]
    beta = weekly_builder.add_fantasy_points(
        _quarterback_week(),
        "QB",
        league="beta",
        filter_qb_workload=False,
    ).iloc[0]

    assert dk["fantasy_pts_pass"] == pytest.approx(22.0)
    assert beta["fantasy_pts_pass"] == pytest.approx(20.0)
    assert dk["fantasy_pts_rush"] == pytest.approx(28.0)
    assert beta["fantasy_pts_rush"] == pytest.approx(28.0)
    assert dk["fantasy_pts"] == pytest.approx(50.0)
    assert beta["fantasy_pts"] == pytest.approx(48.0)


def test_beta_passing_applies_sacks_and_both_yardage_bonus_thresholds():
    rows = pd.DataFrame(
        [
            {"pass_yards_gained_sum": 299, "sack_sum": 0},
            {"pass_yards_gained_sum": 300, "sack_sum": 0},
            {"pass_yards_gained_sum": 400, "sack_sum": 0},
            {"pass_yards_gained_sum": 0, "sack_sum": 2},
        ]
    )

    scored = weekly_builder.add_fantasy_points(
        rows,
        "QB",
        league="beta",
        filter_qb_workload=False,
    )

    assert scored["fantasy_pts_pass"].tolist() == pytest.approx(
        [11.96, 13.0, 19.0, -2.0]
    )


def test_beta_receiving_uses_half_ppr_and_cumulative_yardage_bonuses():
    dk = weekly_builder.add_fantasy_points(
        _receiver_week(),
        "WR",
        league="dk",
    ).iloc[0]
    beta = weekly_builder.add_fantasy_points(
        _receiver_week(),
        "WR",
        league="beta",
    ).iloc[0]

    assert dk["fantasy_pts_rec"] == pytest.approx(39.0)
    assert beta["fantasy_pts_rec"] == pytest.approx(35.0)

    threshold_rows = pd.DataFrame(
        [
            {"rec_complete_pass_sum": 10, "rec_yards_gained_sum": 0},
            {"rec_complete_pass_sum": 0, "rec_yards_gained_sum": 100},
            {"rec_complete_pass_sum": 0, "rec_yards_gained_sum": 200},
        ]
    )
    beta_thresholds = weekly_builder.add_fantasy_points(
        threshold_rows,
        "WR",
        league="beta",
    )
    assert beta_thresholds["fantasy_pts_rec"].tolist() == pytest.approx(
        [5.0, 11.0, 23.0]
    )


def test_load_weekly_points_resolves_monkeypatched_league_at_call_time(
    monkeypatch,
):
    def fake_daily_read(query, database):
        assert database == "FastR_Beta"
        if "FROM QB_Stats" in query:
            return _quarterback_week()
        if "FROM WR_Stats" in query:
            return _receiver_week()
        pos = next(pos for pos in ["RB", "TE"] if f"FROM {pos}_Stats" in query)
        return pd.DataFrame(
            [
                {
                    "player": f"Scoring {pos}",
                    "team": "TST",
                    "season": 2025,
                    "week": 1,
                }
            ]
        )

    monkeypatch.setattr(scoring_config, "LEAGUE", "dk")
    monkeypatch.setattr(weekly_builder, "LEAGUE", "beta")
    monkeypatch.setattr(weekly_builder.dm_daily, "read", fake_daily_read)

    scored = weekly_builder.load_weekly_points(2025)

    qb = scored.loc[scored["pos"].eq("QB")].iloc[0]
    wr = scored.loc[scored["pos"].eq("WR")].iloc[0]
    assert set(scored["scoring_league"]) == {"beta"}
    assert qb["fantasy_pts"] == pytest.approx(48.0)
    assert wr["fantasy_pts"] == pytest.approx(35.0)


def test_template_league_uses_weekly_marker_and_rejects_mismatch(monkeypatch):
    weekly = pd.DataFrame(
        [
            {
                "player": "Scoring Receiver",
                "pos": "WR",
                "season": 2025,
                "week": 1,
                "fantasy_pts": 10.0,
                "managed_fantasy_pts": 10.0,
                "played_week": True,
                "scoring_league": "beta",
            }
        ]
    )
    monkeypatch.setattr(weekly_builder, "LEAGUE", "dk")

    templates = weekly_builder.build_weekly_templates(
        _projection_context(),
        weekly,
    )
    assert templates.loc[0, "league"] == "beta"
    assert templates.loc[0, "template_id"] == (
        weekly_builder.TEMPLATE_ID_LEAGUE_OFFSETS["beta"] + 1
    )
    assert {
        weekly_builder.V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN,
        weekly_builder.V2_TEMPLATE_CENTER_POSITION_COLUMN,
        weekly_builder.V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN,
        weekly_builder.V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN,
    }.issubset(templates.columns)

    with pytest.raises(ValueError, match="does not match requested"):
        weekly_builder.build_weekly_templates(
            _projection_context(),
            weekly,
            league="dk",
        )


def test_cli_exposes_safe_staging_controls(tmp_path):
    staging_db = tmp_path / "Simulation_staging.sqlite3"
    v2_db = tmp_path / "Projection_V2_beta_staging.sqlite3"
    with sqlite3.connect(staging_db) as connection:
        connection.execute("CREATE TABLE staging_marker (value INTEGER)")

    args = weekly_builder.parse_args(
        [
            "--league",
            "beta",
            "--simulation-db",
            str(staging_db),
            "--v2-db",
            str(v2_db),
            "--no-app-sync",
        ]
    )
    assert args.league == "beta"
    assert args.simulation_db == staging_db
    assert args.v2_db == v2_db
    assert args.no_app_sync is True


def test_custom_simulation_db_is_isolated_and_cannot_sync_apps(
    monkeypatch,
    tmp_path,
):
    staging_db = tmp_path / "Simulation_staging.sqlite3"
    with sqlite3.connect(staging_db) as connection:
        connection.execute("CREATE TABLE staging_marker (value INTEGER)")

    monkeypatch.setattr(
        weekly_builder,
        "SIMULATION_DB_PATH",
        weekly_builder.SIMULATION_DB_PATH,
    )
    monkeypatch.setattr(
        weekly_builder,
        "SIMULATION_DB_NAME",
        weekly_builder.SIMULATION_DB_NAME,
    )
    monkeypatch.setattr(
        weekly_builder,
        "simulation_dm",
        weekly_builder.simulation_dm,
    )

    weekly_builder.set_simulation_db(staging_db)
    assert weekly_builder.SIMULATION_DB_PATH == staging_db.resolve()
    assert weekly_builder.db_table_exists("staging_marker")

    with pytest.raises(ValueError, match="requires sync_apps=False"):
        weekly_builder.main(
            league="beta",
            simulation_db=staging_db,
            sync_apps=True,
        )

    with pytest.raises(ValueError, match="requires an explicit v2_database"):
        weekly_builder.main(
            league="beta",
            simulation_db=staging_db,
            sync_apps=False,
        )

    with pytest.raises(ValueError, match="requires a staged V2 database copy"):
        weekly_builder.main(
            league="beta",
            simulation_db=staging_db,
            v2_database=weekly_builder.V2_DATABASES["beta"],
            sync_apps=False,
        )


def test_live_simulation_rejects_staged_v2_database(
    monkeypatch,
    tmp_path,
):
    staged_v2 = (tmp_path / "Projection_V2_beta_staging.sqlite3").resolve()
    staged_v2.touch()
    monkeypatch.setattr(
        weekly_builder,
        "SIMULATION_DB_PATH",
        weekly_builder.DEFAULT_SIMULATION_DB_PATH,
    )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: staged_v2,
    )

    with pytest.raises(
        ValueError,
        match="live Simulation database requires the configured beta V2",
    ):
        weekly_builder.main(
            league="beta",
            v2_database=staged_v2,
            sync_apps=False,
        )


def test_live_simulation_accepts_only_active_league_configured_v2(
    monkeypatch,
):
    configured_beta = Path(
        weekly_builder.V2_DATABASES["beta"]
    ).resolve()
    monkeypatch.setattr(
        weekly_builder,
        "SIMULATION_DB_PATH",
        weekly_builder.DEFAULT_SIMULATION_DB_PATH,
    )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: configured_beta,
    )

    class GuardPassed(RuntimeError):
        pass

    def stop_after_database_guards():
        raise GuardPassed

    monkeypatch.setattr(
        weekly_builder,
        "get_daily_max_template_season",
        stop_after_database_guards,
    )
    with pytest.raises(GuardPassed):
        weekly_builder.main(
            league="beta",
            v2_database=configured_beta,
            sync_apps=False,
        )


def test_v2_override_requires_matching_locked_scoring_objective(tmp_path):
    v2_db = tmp_path / "Projection_V2_staging.sqlite3"
    with sqlite3.connect(v2_db) as connection:
        connection.execute("CREATE TABLE player_identity (player_key TEXT)")
        connection.execute("CREATE TABLE player_aliases (player_key TEXT)")
        connection.execute(
            "CREATE TABLE locked_template_handoff "
            "(player_key TEXT, model_run_id TEXT)"
        )
        connection.execute(
            "CREATE TABLE locked_candidate_runs "
            "(model_run_id TEXT, metadata_json TEXT)"
        )
        connection.execute("CREATE TABLE build_runs (league TEXT)")
        connection.execute("INSERT INTO build_runs VALUES ('dk')")
        connection.execute("INSERT INTO build_runs VALUES ('beta')")
        connection.execute(
            "INSERT INTO locked_template_handoff VALUES ('player-1', 'run-1')"
        )
        connection.execute(
            "INSERT INTO locked_candidate_runs VALUES (?, ?)",
            ("run-1", json.dumps({"scoring_objective": "dk"})),
        )

    with pytest.raises(
        ValueError,
        match="locked handoff scoring objective.*does not match beta",
    ):
        weekly_builder.resolve_v2_database(v2_db, league="beta")

    assert (
        weekly_builder.resolve_v2_database(v2_db, league="dk")
        == v2_db.resolve()
    )

    with sqlite3.connect(v2_db) as connection:
        connection.execute(
            "UPDATE locked_candidate_runs SET metadata_json=?",
            (json.dumps({"scoring_objective": "beta"}),),
        )

    assert (
        weekly_builder.resolve_v2_database(v2_db, league="beta")
        == v2_db.resolve()
    )
    with pytest.raises(
        ValueError,
        match="locked handoff scoring objective.*does not match dk",
    ):
        weekly_builder.resolve_v2_database(v2_db, league="dk")


def test_beta_2018_qb_missing_v2_center_uses_audited_legacy_fallback(
    monkeypatch,
    tmp_path,
):
    v2_db = tmp_path / "Projection_V2_beta_staging.sqlite3"
    _write_center_database(v2_db)
    monkeypatch.setattr(weekly_builder, "LEAGUE", "beta")
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        lambda frame, *args, **kwargs: frame.copy(),
    )

    attached = weekly_builder.attach_locked_v2_historical_centers(
        _historical_center_projection(),
        max_template_season=2018,
        v2_database=v2_db,
    )
    row = attached.iloc[0]

    assert row["historical_pred_fp_per_game"] == pytest.approx(18.5)
    assert row["legacy_historical_pred_fp_per_game"] == pytest.approx(18.5)
    assert pd.isna(row["v2_historical_pred_fp_per_game"])
    assert row["v2_template_center_available"] == 0
    assert (
        row[weekly_builder.V2_TEMPLATE_CENTER_UNAVAILABLE_REASON_COLUMN]
        == weekly_builder.BETA_2018_QB_CENTER_FALLBACK_REASON
    )
    assert row["historical_center_policy"] == "legacy_validated_oos"
    assert row["v2_recenter_promoted"] == 0


def test_beta_2018_qb_fallback_requires_active_quarantine_proof(
    monkeypatch,
    tmp_path,
):
    v2_db = tmp_path / "Projection_V2_beta_staging.sqlite3"
    _write_center_database(v2_db, include_quarantine_proof=False)
    monkeypatch.setattr(weekly_builder, "LEAGUE", "beta")
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        lambda frame, *args, **kwargs: frame.copy(),
    )

    with pytest.raises(ValueError, match="lacks the active FFToday quarantine"):
        weekly_builder.attach_locked_v2_historical_centers(
            _historical_center_projection(),
            max_template_season=2018,
            v2_database=v2_db,
        )


@pytest.mark.parametrize(
    (
        "player",
        "player_key",
        "season",
        "template_position",
        "center_position",
        "reason",
    ),
    [
        (
            "Cordarrelle Patterson",
            "b16d3ba0-39d4-5a4b-bca8-ad15e147c96b",
            2019,
            "WR",
            "RB",
            "canonical_hybrid_role_shift:cordarrelle_patterson",
        ),
        (
            "Cordarrelle Patterson",
            "b16d3ba0-39d4-5a4b-bca8-ad15e147c96b",
            2021,
            "WR",
            "RB",
            "canonical_hybrid_role_shift:cordarrelle_patterson",
        ),
        (
            "Ty Montgomery",
            "2f3a5f36-ad51-527b-8fdc-ca0a5e431ad6",
            2022,
            "RB",
            "WR",
            "canonical_hybrid_role_shift:ty_montgomery",
        ),
    ],
)
def test_governed_hybrid_v2_center_positions_are_retained_and_audited(
    monkeypatch,
    tmp_path,
    player,
    player_key,
    season,
    template_position,
    center_position,
    reason,
):
    v2_db = tmp_path / "Projection_V2_staging.sqlite3"
    _write_center_database(
        v2_db,
        center=19.0,
        center_available=1,
        player_key=player_key,
        center_season=season,
        center_position=center_position,
    )
    monkeypatch.setattr(weekly_builder, "LEAGUE", "dk")
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        lambda frame, *args, **kwargs: frame.copy(),
    )

    attached = weekly_builder.attach_locked_v2_historical_centers(
        _historical_center_projection(
            season=season,
            pos=template_position,
            player=player,
            player_key=player_key,
        ),
        max_template_season=season,
        v2_database=v2_db,
    )
    row = attached.iloc[0]

    assert row[weekly_builder.V2_TEMPLATE_CENTER_POSITION_COLUMN] == (
        center_position
    )
    assert (
        row[weekly_builder.V2_TEMPLATE_CENTER_POSITION_MISMATCH_COLUMN]
        == 1
    )
    assert (
        row[
            weekly_builder.V2_TEMPLATE_CENTER_POSITION_MISMATCH_REASON_COLUMN
        ]
        == reason
    )
    assert row["historical_pred_fp_per_game"] == pytest.approx(18.5)


def test_ungoverned_v2_center_position_mismatch_fails_closed(
    monkeypatch,
    tmp_path,
):
    v2_db = tmp_path / "Projection_V2_staging.sqlite3"
    _write_center_database(
        v2_db,
        center=19.0,
        center_available=1,
        center_position="RB",
    )
    monkeypatch.setattr(weekly_builder, "LEAGUE", "dk")
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        lambda frame, *args, **kwargs: frame.copy(),
    )

    with pytest.raises(ValueError, match="center positions are inconsistent"):
        weekly_builder.attach_locked_v2_historical_centers(
            _historical_center_projection(pos="WR"),
            max_template_season=2018,
            v2_database=v2_db,
        )


@pytest.mark.parametrize(
    ("center", "center_available"),
    [
        (None, 1),
        (19.0, 0),
    ],
)
def test_locked_v2_center_value_and_availability_must_agree(
    monkeypatch,
    tmp_path,
    center,
    center_available,
):
    v2_db = tmp_path / "Projection_V2_beta_staging.sqlite3"
    _write_center_database(
        v2_db,
        center=center,
        center_available=center_available,
    )
    monkeypatch.setattr(weekly_builder, "LEAGUE", "beta")
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        lambda frame, *args, **kwargs: frame.copy(),
    )

    with pytest.raises(ValueError, match="center availability is inconsistent"):
        weekly_builder.attach_locked_v2_historical_centers(
            _historical_center_projection(),
            max_template_season=2018,
            v2_database=v2_db,
        )


@pytest.mark.parametrize(
    ("season", "pos", "league"),
    [
        (2019, "QB", "beta"),
        (2018, "WR", "beta"),
        (2018, "QB", "dk"),
    ],
)
def test_missing_v2_center_fails_outside_declared_beta_2018_qb_slice(
    monkeypatch,
    tmp_path,
    season,
    pos,
    league,
):
    v2_db = tmp_path / "Projection_V2_staging.sqlite3"
    _write_center_database(v2_db)
    if season != 2018:
        with sqlite3.connect(v2_db) as connection:
            connection.execute(
                "UPDATE locked_template_handoff SET season=?",
                (season,),
            )
    if pos != "QB":
        with sqlite3.connect(v2_db) as connection:
            connection.execute(
                "UPDATE locked_template_handoff SET position=?",
                (pos,),
            )
    monkeypatch.setattr(weekly_builder, "LEAGUE", league)
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        lambda frame, *args, **kwargs: frame.copy(),
    )

    with pytest.raises(ValueError, match="point-center coverage is incomplete"):
        weekly_builder.attach_locked_v2_historical_centers(
            _historical_center_projection(season=season, pos=pos),
            max_template_season=season,
            v2_database=v2_db,
        )


def test_beta_2018_qb_fallback_requires_a_joined_locked_center_row(
    monkeypatch,
    tmp_path,
):
    v2_db = tmp_path / "Projection_V2_beta_staging.sqlite3"
    _write_center_database(v2_db, include_center_row=False)
    monkeypatch.setattr(weekly_builder, "LEAGUE", "beta")
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        lambda frame, *args, **kwargs: frame.copy(),
    )

    with pytest.raises(
        ValueError,
        match="point-center handoff coverage is incomplete",
    ):
        weekly_builder.attach_locked_v2_historical_centers(
            _historical_center_projection(),
            max_template_season=2018,
            v2_database=v2_db,
        )
