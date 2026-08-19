import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from Scripts.Modeling import s4_Best_Ball_Weekly as weekly_builder
import config as scoring_config


def test_adp_audit_joins_the_published_player_key_directly(monkeypatch):
    player_map = pd.DataFrame(
        {
            "player_key": ["canonical-key"],
            "player": ["Canonical Display"],
            "pos": ["WR"],
            "team": ["TST"],
            "year": [weekly_builder.YEAR],
            "version": ["dk"],
            "dataset": [weekly_builder.PRED_VERSION],
            "pred_fp_per_game": [15.0],
            "avg_pick": [99.0],
        }
    )
    published_adp = pd.DataFrame(
        {
            "player_key": ["canonical-key"],
            "avg_adp_player": ["Provider Label With No Governed Alias"],
            "year": [weekly_builder.YEAR],
            "league": ["dk"],
            "avg_adp_pick": [12.5],
            "avg_adp_std_dev": [1.0],
            "avg_adp_min_pick": [10.0],
            "avg_adp_max_pick": [15.0],
            "avg_adp_year_exp_app_match": [2.0],
            "avg_adp_key_match_method": ["alias_confirmed_unique"],
        }
    )
    monkeypatch.setattr(weekly_builder, "LEAGUE", "dk")
    monkeypatch.setattr(
        weekly_builder.simulation_dm,
        "read",
        lambda *_args, **_kwargs: published_adp.copy(),
    )

    audit = weekly_builder.build_adp_audit(player_map)

    assert audit.loc[0, "avg_adp_pick"] == 12.5
    assert audit.loc[0, "avg_adp_player"] == (
        "Provider Label With No Governed Alias"
    )
    assert audit.loc[0, "avg_adp_key_match_method"] == (
        "alias_confirmed_unique"
    )
    assert audit.loc[0, "missing_avg_adp_match"] == 0


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
    nv = weekly_builder.add_fantasy_points(
        _quarterback_week(),
        "QB",
        league="nv",
        filter_qb_workload=False,
    ).iloc[0]

    assert dk["fantasy_pts_pass"] == pytest.approx(22.0)
    assert beta["fantasy_pts_pass"] == pytest.approx(20.0)
    assert nv["fantasy_pts_pass"] == pytest.approx(19.0)
    assert dk["fantasy_pts_rush"] == pytest.approx(28.0)
    assert beta["fantasy_pts_rush"] == pytest.approx(28.0)
    assert nv["fantasy_pts_rush"] == pytest.approx(28.0)
    assert dk["fantasy_pts"] == pytest.approx(50.0)
    assert beta["fantasy_pts"] == pytest.approx(48.0)
    assert nv["fantasy_pts"] == pytest.approx(47.0)


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


def test_zero_active_managed_qb_uses_conditional_center(monkeypatch):
    monkeypatch.setattr(weekly_builder, "WEEKS", list(range(1, 17)))
    projection = _projection_context()
    projection.loc[0, ["player", "pos"]] = ["Cameo Quarterback", "QB"]
    projection.loc[0, "historical_pred_fp_per_game"] = 0.01
    projection.loc[0, "v2_historical_pred_fp_per_game"] = 10.0
    projection.loc[0, "v2_template_center_available"] = 1
    projection["v2_point_center_source"] = projection[
        "v2_point_center_source"
    ].astype("object")
    projection.loc[0, "v2_point_center_source"] = "fixture_conditional"
    weekly = pd.DataFrame(
        [
            {
                "player": "Cameo Quarterback",
                "pos": "QB",
                "season": 2025,
                "week": 1,
                "fantasy_pts": None,
                "managed_fantasy_pts": 5.0,
                "played_week": True,
                "scoring_league": "nv",
            }
        ]
    )

    template = weekly_builder.build_weekly_templates(
        projection,
        weekly,
        league="nv",
    ).iloc[0]

    assert template["active_games"] == 0
    assert template["played_games"] == 1
    assert template["active_ppg_resid"] == pytest.approx(-0.01)
    assert template[weekly_builder.MANAGED_PROFILE_PPG_COLUMN] == pytest.approx(10.0)
    assert template[weekly_builder.MANAGED_RESIDUAL_CENTER_COLUMN] == pytest.approx(10.0)
    assert template[weekly_builder.MANAGED_ACTIVE_PPG_RESID_COLUMN] == pytest.approx(-10.0)
    assert template[weekly_builder.MANAGED_CENTER_POLICY_COLUMN] == "v2_conditional"
    assert template["managed_week_1"] == pytest.approx(0.5)
    assert template["managed_profile_total"] == pytest.approx(0.5)
    assert template["profile_total"] == pytest.approx(0.0)


def test_template_excludes_governed_unavailable_scoring_context():
    projection = _projection_context()
    projection["scoring_context_available"] = 0
    projection["scoring_context_unavailable_reason"] = (
        weekly_builder.BETA_2018_QB_CENTER_FALLBACK_REASON
    )
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

    template = weekly_builder.build_weekly_templates(
        projection,
        weekly,
        league="beta",
    ).iloc[0]

    assert template["template_eligible"] == 0
    assert template["template_exclusion_reason"] == (
        "scoring_context_unavailable:"
        + weekly_builder.BETA_2018_QB_CENTER_FALLBACK_REASON
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


def test_nffc_uses_governed_17_week_modern_donor_contract():
    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("nffc")
        assert weekly_builder.WEEK_COUNT == 17
        assert weekly_builder.WEEKS == list(range(1, 18))
        assert weekly_builder.TEMPLATE_SEASON_MIN == 2021
    finally:
        weekly_builder.set_active_league(original_league)


def test_modern_nffc_audit_does_not_require_out_of_era_exclusions():
    player_pool_audit = pd.DataFrame(
        columns=[
            "player",
            "missing_template_pool",
            "template_pool_below_min",
        ]
    )
    template_audit = pd.DataFrame(
        [
            {
                "player": "Modern Donor",
                "pos": "RB",
                "season": 2021,
                "template_eligible": 1,
                "template_exclusion_reason": "",
                "played_mask_mismatch": False,
                "active_exceeds_played": False,
                "non_qb_played_active_mismatch": False,
            }
        ]
    )

    weekly_builder.validate_weekly_template_audits(
        player_pool_audit,
        template_audit,
    )


def test_beta_audit_accepts_only_governed_2018_qb_context_exclusions():
    player_pool_audit = pd.DataFrame(
        columns=[
            "player",
            "missing_template_pool",
            "template_pool_below_min",
        ]
    )
    unavailable_reason = (
        "scoring_context_unavailable:"
        + weekly_builder.BETA_2018_QB_CENTER_FALLBACK_REASON
    )
    template_audit = pd.DataFrame(
        [
            {
                "player": "Quarantined Quarterback",
                "pos": "QB",
                "season": 2018,
                "template_eligible": 0,
                "template_exclusion_reason": unavailable_reason,
                "played_mask_mismatch": False,
                "active_exceeds_played": False,
                "non_qb_played_active_mismatch": False,
            },
            {
                "player": "Le'Veon Bell",
                "pos": "RB",
                "season": 2018,
                "template_eligible": 0,
                "template_exclusion_reason": "contract_holdout",
                "played_mask_mismatch": False,
                "active_exceeds_played": False,
                "non_qb_played_active_mismatch": False,
            },
        ]
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("beta")
        weekly_builder.validate_weekly_template_audits(
            player_pool_audit,
            template_audit,
        )

        invalid = template_audit.copy()
        invalid.loc[invalid["pos"].eq("QB"), "pos"] = "WR"
        with pytest.raises(ValueError, match="outside beta 2018 QB"):
            weekly_builder.validate_weekly_template_audits(
                player_pool_audit,
                invalid,
            )
    finally:
        weekly_builder.set_active_league(original_league)


def test_weekly_rebuild_drops_prior_year_active_league_artifacts():
    pool_rows = pd.DataFrame(
        [
            {
                "pool_year": 2025,
                "pool_version": "nffc",
                "pool_dataset": "final_ensemble",
            },
            {
                "pool_year": 2026,
                "pool_version": "nffc",
                "pool_dataset": "final_ensemble",
            },
            {
                "pool_year": 2025,
                "pool_version": "dk",
                "pool_dataset": "final_ensemble",
            },
        ]
    )
    map_rows = pd.DataFrame(
        [
            {"year": 2025, "version": "nffc", "dataset": "final_ensemble"},
            {"year": 2026, "version": "nffc", "dataset": "final_ensemble"},
            {"year": 2025, "version": "dk", "dataset": "final_ensemble"},
        ]
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("nffc")
        retained_pool = pool_rows[
            weekly_builder.keep_not_current_pool_slice(pool_rows)
        ]
        retained_map = map_rows[
            weekly_builder.keep_not_current_prediction_slice(map_rows)
        ]
    finally:
        weekly_builder.set_active_league(original_league)

    assert retained_pool[["pool_year", "pool_version"]].to_dict(
        "records"
    ) == [{"pool_year": 2025, "pool_version": "dk"}]
    assert retained_map[["year", "version"]].to_dict("records") == [
        {"year": 2025, "version": "dk"}
    ]


def _current_prediction(player_key, player, *, ppg=8.0):
    return {
        "player_key": player_key,
        "player": player,
        "pos": "WR",
        "year": 2026,
        "version": "dk",
        "dataset": "final_ensemble",
        "pred_fp_per_game": ppg,
    }


def _current_context(player_key, *, source, team="WAS"):
    return {
        "player_key": player_key,
        "pos": "WR",
        "year": 2026,
        "team": team,
        "current_avg_proj_points": 144.0,
        "avg_pick": 170.0,
        "year_exp": 7.0,
        "current_context_source": source,
        "current_context_match_method": "alias_confirmed_unique",
        "current_context_missing_optional_fields": "",
    }


def _scoring_match_context(
    player_key,
    *,
    source,
    total_points,
    projection_ppg,
    pass_points,
    rush_points,
    receiving_points,
    team_qb1_points,
    projection_std_points,
):
    return {
        **_current_context(player_key, source=source),
        "current_avg_proj_points": total_points,
        "avg_proj_points": total_points,
        "avg_proj_pass_points": pass_points,
        "avg_proj_rush_points": rush_points,
        "avg_proj_rec_points": receiving_points,
        "qb_avg_proj_pass_points": team_qb1_points,
        "std_proj_points": projection_std_points,
        "std_pos_rank": 2.0,
        "match_projection_ppg_scaled": (
            projection_ppg / weekly_builder.PROJECTION_PPG_SCALE
        ),
    }


def _write_v2_scoring_context_database(
    path,
    *,
    season,
    total_points=204.0,
    league="nffc",
    receiver_shares=(0.10, 0.30, 0.60),
):
    feature_run_id = f"{league}-feature-run"
    model_run_id = f"{league}-scoring-run"
    pass_share, rush_share, receiving_share = receiver_shares
    feature = pd.DataFrame(
        [
            {
                "player_key": "scoring-key",
                "display_name": "Scoring Receiver",
                "season": season,
                "position": "WR",
                "team": "WAS",
                "league": league,
                "scoring_hash": weekly_builder.scoring_hash(league),
                "run_id": feature_run_id,
                "feature_cutoff_season": season - 1,
                "preseason_source_season": season,
                "expert_points_median": total_points,
                "expert_ppg_team_game_median": 12.0,
                "expert_ppg_team_game_std": 1.5,
                "expert_points_iqr": 20.0,
                "adp_median": 45.0,
                "year_exp": 3.0,
                "projected_pass_point_share": pass_share,
                "projected_rush_point_share": rush_share,
                "projected_receiving_point_share": receiving_share,
                "team_qb1_ppg": 22.0,
            },
            {
                "player_key": "team-qb-key",
                "display_name": "Scoring Quarterback",
                "season": season,
                "position": "QB",
                "team": "WAS",
                "league": league,
                "scoring_hash": weekly_builder.scoring_hash(league),
                "run_id": feature_run_id,
                "feature_cutoff_season": season - 1,
                "preseason_source_season": season,
                "expert_points_median": 374.0,
                "expert_ppg_team_game_median": 22.0,
                "expert_ppg_team_game_std": 2.0,
                "expert_points_iqr": 30.0,
                "adp_median": 60.0,
                "year_exp": 4.0,
                "projected_pass_point_share": 0.75,
                "projected_rush_point_share": 0.25,
                "projected_receiving_point_share": 0.0,
                "team_qb1_ppg": 22.0,
            },
        ]
    )
    handoff = pd.DataFrame(
        [
            {
                "model_run_id": model_run_id,
                "player_key": "scoring-key",
                "season": season,
                "position": "WR",
                "historical_pred_fp_per_game": 11.5,
                "point_center_source": "locked_nffc_fixture",
                "template_center_available": 1,
            }
        ]
    )
    projection_values = pd.DataFrame(
        [
            {
                "player_key": "scoring-key",
                "season": season,
                "provider": "provider_a",
                "position": "WR",
                "configured_points_complete": 1,
                "provider_projected_points": total_points,
                "run_id": feature_run_id,
            },
            {
                "player_key": "scoring-key",
                "season": season,
                "provider": "provider_b",
                "position": "WR",
                "configured_points_complete": 1,
                "provider_projected_points": total_points - 24.0,
                "run_id": feature_run_id,
            },
            {
                "player_key": "other-wr-key",
                "season": season,
                "provider": "provider_a",
                "position": "WR",
                "configured_points_complete": 1,
                "provider_projected_points": total_points - 12.0,
                "run_id": feature_run_id,
            },
            {
                "player_key": "other-wr-key",
                "season": season,
                "provider": "provider_b",
                "position": "WR",
                "configured_points_complete": 1,
                "provider_projected_points": total_points + 12.0,
                "run_id": feature_run_id,
            },
            {
                "player_key": "team-qb-key",
                "season": season,
                "provider": "provider_a",
                "position": "QB",
                "configured_points_complete": 1,
                "provider_projected_points": 374.0,
                "run_id": feature_run_id,
            },
            {
                "player_key": "team-qb-key",
                "season": season,
                "provider": "provider_b",
                "position": "QB",
                "configured_points_complete": 1,
                "provider_projected_points": 350.0,
                "run_id": feature_run_id,
            },
        ]
    )
    with sqlite3.connect(path) as connection:
        feature.to_sql(
            "player_season_features",
            connection,
            if_exists="replace",
            index=False,
        )
        handoff.to_sql(
            "locked_template_handoff",
            connection,
            if_exists="replace",
            index=False,
        )
        projection_values.to_sql(
            "player_season_projection_values",
            connection,
            if_exists="replace",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "model_run_id": model_run_id,
                    "feature_run_id": feature_run_id,
                }
            ]
        ).to_sql(
            "locked_candidate_runs",
            connection,
            if_exists="replace",
            index=False,
        )


def test_nffc_v2_current_context_uses_scoring_components_and_17_week_ppg(
    monkeypatch,
    tmp_path,
):
    v2_db = tmp_path / "Projection_V2_nffc_staging.sqlite3"
    _write_v2_scoring_context_database(
        v2_db,
        season=weekly_builder.YEAR,
    )
    published_adp = pd.DataFrame(
        [
            {
                    "player_key": "scoring-key",
                    "year": weekly_builder.YEAR,
                    "adp_team": pd.NA,
                    "adp_avg_pick": 40.0,
                "adp_year_exp": 3.0,
            }
        ]
    )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "load_published_current_adp_context",
        lambda: published_adp.copy(),
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("nffc")
        context = weekly_builder.load_v2_current_player_context(
            v2_database=v2_db,
            selected_player_keys={"scoring-key"},
        ).iloc[0]
    finally:
        weekly_builder.set_active_league(original_league)

    assert context["current_avg_proj_points"] == pytest.approx(204.0)
    assert context["current_projection_ppg"] == pytest.approx(12.0)
    assert context["match_projection_ppg_scaled"] == pytest.approx(1.2)
    assert context["avg_proj_pass_points"] == pytest.approx(20.4)
    assert context["avg_proj_rush_points"] == pytest.approx(61.2)
    assert context["avg_proj_rec_points"] == pytest.approx(122.4)
    assert context["qb_avg_proj_pass_points"] == pytest.approx(280.5)
    assert context["std_proj_points"] == pytest.approx(25.5)


@pytest.mark.parametrize("league", ["dk", "beta", "nv"])
def test_non_nffc_v2_current_context_derives_qb_passing_component_before_filter(
    monkeypatch,
    tmp_path,
    league,
):
    v2_db = tmp_path / f"Projection_V2_{league}_staging.sqlite3"
    _write_v2_scoring_context_database(
        v2_db,
        season=weekly_builder.YEAR,
    )
    # Exercise a legacy V2 team spelling on the selected receiver while the
    # QB1 row uses the modern spelling. The QB is intentionally excluded from
    # selected_player_keys so derivation must precede the production filter.
    with sqlite3.connect(v2_db) as connection:
        connection.execute(
            """
            UPDATE player_season_features
            SET team=CASE position
                    WHEN 'WR' THEN 'TAM'
                    ELSE 'TB'
                END,
                team_qb1_ppg=NULL
            """
        )
    published_adp = pd.DataFrame(
        [
            {
                    "player_key": "scoring-key",
                    "year": weekly_builder.YEAR,
                    "adp_team": pd.NA,
                    "adp_avg_pick": 40.0,
                "adp_year_exp": 3.0,
            }
        ]
    )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )
    monkeypatch.setattr(
        weekly_builder,
        "load_published_current_adp_context",
        lambda: published_adp.copy(),
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league(league)
        context = weekly_builder.load_v2_current_player_context(
            v2_database=v2_db,
            selected_player_keys={"scoring-key"},
            scoring_matched_context=False,
        ).iloc[0]
    finally:
        weekly_builder.set_active_league(original_league)

    # A missing precomputed team_qb1_ppg is filled from the selected QB1 row;
    # its independently derived passing component remains authoritative.
    assert context["qb_avg_proj_pass_points"] == pytest.approx(280.5)
    assert context["team_qb1_ppg"] == pytest.approx(22.0)
    if league in weekly_builder.AUCTION_ETR_LEAGUES:
        assert context["avg_pick"] == pytest.approx(45.0)
        assert context["current_adp_source"] == (
            "v2_canonical_adp_family_consensus"
        )
    else:
        assert context["avg_pick"] == pytest.approx(40.0)
        assert context["current_adp_source"] == "canonical_avg_adps"


def _model_input_scoring_projection(*, season=2025):
    return pd.DataFrame(
        [
            {
                "player": "Scoring Receiver",
                "player_key": "scoring-key",
                "pos": "WR",
                "team": "WAS",
                "season": season,
                "avg_proj_points": 100.0,
                "preseason_proj_ppg": 6.25,
                "avg_proj_pass_points": 1.0,
                "avg_proj_rush_points": 19.0,
                "avg_proj_rec_points": 80.0,
                "qb_avg_proj_pass_points": 250.0,
                "std_proj_points": 5.0,
            }
        ]
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "projected_receiving_point_share",
            None,
            "invalid projected_receiving_point_share",
        ),
        (
            "projected_receiving_point_share",
            0.50,
            "component shares do not sum to one",
        ),
    ],
)
def test_nffc_scored_context_rejects_incomplete_positive_total_shares(
    monkeypatch,
    tmp_path,
    column,
    value,
    message,
):
    v2_db = tmp_path / "Projection_V2_nffc_staging.sqlite3"
    _write_v2_scoring_context_database(v2_db, season=2025)
    with sqlite3.connect(v2_db) as connection:
        connection.execute(
            f'UPDATE player_season_features SET "{column}"=?',
            (value,),
        )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("nffc")
        with pytest.raises(ValueError, match=message):
            weekly_builder.apply_v2_scored_projection_context(
                _model_input_scoring_projection(),
                v2_database=v2_db,
                season_column="season",
            )
    finally:
        weekly_builder.set_active_league(original_league)


def test_nffc_scored_context_rejects_total_ppg_contract_mismatch(
    monkeypatch,
    tmp_path,
):
    v2_db = tmp_path / "Projection_V2_nffc_staging.sqlite3"
    _write_v2_scoring_context_database(v2_db, season=2025)
    with sqlite3.connect(v2_db) as connection:
        connection.execute(
            "UPDATE player_season_features "
            "SET expert_ppg_team_game_median=11.5"
        )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("nffc")
        with pytest.raises(
            ValueError,
            match="total and team-game PPG disagree",
        ):
            weekly_builder.apply_v2_scored_projection_context(
                _model_input_scoring_projection(),
                v2_database=v2_db,
                season_column="season",
            )
    finally:
        weekly_builder.set_active_league(original_league)


def test_nffc_scored_context_rejects_missing_assigned_team_qb(
    monkeypatch,
    tmp_path,
):
    v2_db = tmp_path / "Projection_V2_nffc_staging.sqlite3"
    _write_v2_scoring_context_database(v2_db, season=2025)
    with sqlite3.connect(v2_db) as connection:
        connection.execute(
            "UPDATE player_season_features SET team_qb1_ppg=NULL"
        )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("nffc")
        with pytest.raises(
            ValueError,
            match="lacks valid team-QB context",
        ):
            weekly_builder.apply_v2_scored_projection_context(
                _model_input_scoring_projection(),
                v2_database=v2_db,
                season_column="season",
            )
    finally:
        weekly_builder.set_active_league(original_league)


def _exercise_historical_scoring_context(
    monkeypatch,
    tmp_path,
    *,
    league,
    scoring_matched_context=None,
    scoring_matched_fallback_center=None,
    validation_ppg=6.0,
    receiver_shares=(0.10, 0.30, 0.60),
):
    season = 2025
    v2_db = tmp_path / f"Projection_V2_{league}_staging.sqlite3"
    _write_v2_scoring_context_database(
        v2_db,
        season=season,
        league=league,
        receiver_shares=receiver_shares,
    )
    projection_columns = [
        "player",
        "pos",
        "team",
        "season",
        "avg_proj_points",
        "avg_pick",
        "year_exp",
        "avg_proj_pass_points",
        "avg_proj_rush_points",
        "avg_proj_rec_points",
        "qb_avg_proj_pass_points",
        "std_proj_points",
        "std_pos_rank",
    ]
    model_input_row = pd.DataFrame(
        [
            {
                "player": "Scoring Receiver",
                "pos": "WR",
                "team": "WAS",
                "season": season,
                "avg_proj_points": 100.0,
                "avg_pick": 50.0,
                "year_exp": 3.0,
                "avg_proj_pass_points": 1.0,
                "avg_proj_rush_points": 19.0,
                "avg_proj_rec_points": 80.0,
                "qb_avg_proj_pass_points": 250.0,
                "std_proj_points": 5.0,
                "std_pos_rank": 2.0,
            }
        ]
    )

    monkeypatch.setattr(
        weekly_builder,
        "projection_select_cols",
        lambda *_args, **_kwargs: projection_columns,
    )

    def fake_model_input_read(query, _database):
        if f"FROM WR_{weekly_builder.YEAR}_ProjOnly" in query:
            return model_input_row.copy()
        return pd.DataFrame(columns=projection_columns)

    monkeypatch.setattr(weekly_builder.dm, "read", fake_model_input_read)
    monkeypatch.setattr(
        weekly_builder,
        "attach_uncapped_template_experience",
        lambda frame, season_col: frame.copy(),
    )
    monkeypatch.setattr(
        weekly_builder,
        "load_validation_ensemble_predictions",
        lambda _max_season: pd.DataFrame(
            [
                {
                    "player": "Scoring Receiver",
                    "season": season,
                    "pos": "WR",
                    "validation_pred_fp_per_game": validation_ppg,
                    "validation_ensemble_sources": "dk_scaled_fixture",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        weekly_builder,
        "resolve_v2_database",
        lambda database=None, league=None: v2_db,
    )

    def attach_fixture_key(frame, *_args, **_kwargs):
        output = frame.copy()
        output["player_key"] = "scoring-key"
        output["player_key_match_method"] = "fixture"
        return output

    monkeypatch.setattr(
        weekly_builder,
        "attach_v2_player_keys",
        attach_fixture_key,
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league(league)
        return weekly_builder.load_historical_projection_context(
            season,
            v2_database=v2_db,
            scoring_matched_context=scoring_matched_context,
            scoring_matched_fallback_center=(
                scoring_matched_fallback_center
            ),
        ).iloc[0]
    finally:
        weekly_builder.set_active_league(original_league)


def test_nffc_historical_matching_uses_v2_scoring_context(
    monkeypatch,
    tmp_path,
):
    context = _exercise_historical_scoring_context(
        monkeypatch,
        tmp_path,
        league="nffc",
    )

    assert context["avg_proj_points"] == pytest.approx(204.0)
    assert context["preseason_proj_ppg"] == pytest.approx(12.0)
    assert context["match_projection_ppg_scaled"] == pytest.approx(1.2)
    assert context["avg_proj_pass_points"] == pytest.approx(20.4)
    assert context["avg_proj_rush_points"] == pytest.approx(61.2)
    assert context["avg_proj_rec_points"] == pytest.approx(122.4)
    assert context["qb_avg_proj_pass_points"] == pytest.approx(280.5)
    assert context["std_proj_points"] == pytest.approx(25.5)


def test_beta_historical_scoring_context_allows_signed_components(
    monkeypatch,
    tmp_path,
):
    context = _exercise_historical_scoring_context(
        monkeypatch,
        tmp_path,
        league="beta",
        scoring_matched_context=True,
        scoring_matched_fallback_center=False,
        receiver_shares=(-0.10, 0.50, 0.60),
    )

    assert context["avg_proj_points"] == pytest.approx(204.0)
    assert context["preseason_proj_ppg"] == pytest.approx(12.0)
    assert context["historical_pred_fp_per_game"] == pytest.approx(6.0)
    assert context["match_projection_ppg_scaled"] == pytest.approx(0.6)
    assert context["avg_proj_pass_points"] == pytest.approx(-20.4)
    assert context["avg_proj_rush_points"] == pytest.approx(102.0)
    assert context["avg_proj_rec_points"] == pytest.approx(122.4)
    assert context["std_proj_points"] == pytest.approx(25.5)
    assert context["std_pos_rank"] == pytest.approx(0.5)
    assert context["rank_disagreement_scaled"] == pytest.approx(0.5)
    assert context["model_input_std_pos_rank"] == pytest.approx(2.0)
    assert context["projection_context_source"] == (
        "v2_beta_scoring_matched_preseason"
    )


def test_nv_historical_scoring_context_uses_nv_expert_consensus(
    monkeypatch,
    tmp_path,
):
    context = _exercise_historical_scoring_context(
        monkeypatch,
        tmp_path,
        league="nv",
        scoring_matched_context=True,
        scoring_matched_fallback_center=False,
        receiver_shares=(-0.10, 0.50, 0.60),
    )

    assert context["historical_pred_fp_per_game"] == pytest.approx(12.0)
    assert context["historical_projection_source"] == (
        "v2_nv_expert_consensus"
    )
    assert context["historical_center_policy"] == (
        "nv_scored_expert_consensus"
    )
    assert context["avg_proj_pass_points"] == pytest.approx(-20.4)
    assert context["projection_context_source"] == (
        "v2_nv_scoring_matched_preseason"
    )


def test_beta_historical_scoring_context_replaces_only_dk_fallback_center(
    monkeypatch,
    tmp_path,
):
    context = _exercise_historical_scoring_context(
        monkeypatch,
        tmp_path,
        league="beta",
        validation_ppg=None,
    )

    assert context["legacy_historical_pred_fp_per_game"] == pytest.approx(
        6.25
    )
    assert context["historical_pred_fp_per_game"] == pytest.approx(12.0)
    assert context["historical_projection_source"] == (
        "v2_beta_expert_consensus_fallback"
    )
    assert context["historical_center_policy"] == (
        "beta_scored_expert_fallback"
    )


@pytest.mark.parametrize("league", ["dk", "beta"])
def test_non_nffc_historical_matching_keeps_model_input_context(
    monkeypatch,
    tmp_path,
    league,
):
    context = _exercise_historical_scoring_context(
        monkeypatch,
        tmp_path,
        league=league,
        scoring_matched_context=False,
    )

    assert context["avg_proj_points"] == pytest.approx(100.0)
    assert context["preseason_proj_ppg"] == pytest.approx(6.25)
    assert context["match_projection_ppg_scaled"] == pytest.approx(0.6)
    assert context["avg_proj_pass_points"] == pytest.approx(1.0)
    assert context["avg_proj_rush_points"] == pytest.approx(19.0)
    assert context["avg_proj_rec_points"] == pytest.approx(80.0)
    assert context["qb_avg_proj_pass_points"] == pytest.approx(250.0)
    assert context["std_proj_points"] == pytest.approx(5.0)


def test_nffc_current_join_prefers_v2_scoring_context_over_model_inputs():
    prediction = _current_prediction("scoring-key", "Scoring Receiver")
    prediction["version"] = "nffc"
    model_context = pd.DataFrame(
        [
            _scoring_match_context(
                "scoring-key",
                source="model_inputs_projection_context",
                total_points=100.0,
                projection_ppg=6.0,
                pass_points=1.0,
                rush_points=19.0,
                receiving_points=80.0,
                team_qb1_points=250.0,
                projection_std_points=5.0,
            )
        ]
    )
    v2_context = pd.DataFrame(
        [
            _scoring_match_context(
                "scoring-key",
                source="v2_player_season_features_scoring_context",
                total_points=204.0,
                projection_ppg=12.0,
                pass_points=20.4,
                rush_points=61.2,
                receiving_points=122.4,
                team_qb1_points=374.0,
                projection_std_points=25.5,
            )
        ]
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league("nffc")
        attached = weekly_builder.attach_current_context_by_player_key(
            pd.DataFrame([prediction]),
            model_context,
            v2_context,
        ).iloc[0]
    finally:
        weekly_builder.set_active_league(original_league)

    assert attached["current_avg_proj_points"] == pytest.approx(204.0)
    assert attached["avg_proj_points"] == pytest.approx(204.0)
    assert attached["avg_proj_pass_points"] == pytest.approx(20.4)
    assert attached["avg_proj_rush_points"] == pytest.approx(61.2)
    assert attached["avg_proj_rec_points"] == pytest.approx(122.4)
    assert attached["qb_avg_proj_pass_points"] == pytest.approx(374.0)
    assert attached["std_proj_points"] == pytest.approx(25.5)
    assert attached["match_projection_ppg_scaled"] == pytest.approx(1.2)


def test_beta_scoring_context_preserves_signed_team_qb_passing_component():
    frame = pd.DataFrame(
        [
            {
                "player": "Receiver",
                "pos": "WR",
                "team": "PIT",
                "season": 2026,
                "avg_proj_points": 180.0,
                "historical_pred_fp_per_game": 11.0,
                "projection_rank_pct": 0.5,
                "avg_pick": 50.0,
                "year_exp": 3.0,
                "avg_proj_rec_points": 180.0,
                "qb_avg_proj_pass_points": -5.0,
            },
            {
                "player": "Rushing Starter",
                "pos": "QB",
                "team": "PIT",
                "season": 2026,
                "avg_proj_points": 300.0,
                "historical_pred_fp_per_game": 18.0,
                "projection_rank_pct": 0.5,
                "avg_pick": 100.0,
                "year_exp": 4.0,
                "avg_proj_pass_points": -5.0,
                "avg_proj_rush_points": 305.0,
                "qb_avg_proj_pass_points": -5.0,
            },
            {
                "player": "Passing Backup",
                "pos": "QB",
                "team": "PIT",
                "season": 2026,
                "avg_proj_points": 100.0,
                "historical_pred_fp_per_game": 6.0,
                "projection_rank_pct": 1.0,
                "avg_pick": 200.0,
                "year_exp": 2.0,
                "avg_proj_pass_points": 20.0,
                "avg_proj_rush_points": 80.0,
                "qb_avg_proj_pass_points": -5.0,
            },
        ]
    )

    output = weekly_builder.add_template_match_features(
        frame,
        group_cols=["season", "pos"],
        rank_pct_col="projection_rank_pct",
        total_points_col="avg_proj_points",
        projection_ppg_col="historical_pred_fp_per_game",
        preserve_signed_team_qb_context=True,
    )

    receiver = output[output.player.eq("Receiver")].iloc[0]
    assert receiver["team_qb_pass_points"] == pytest.approx(-5.0)


@pytest.mark.parametrize("league", ["beta", "nv"])
def test_auction_current_join_uses_promoted_v2_scoring_context(league):
    prediction = _current_prediction("scoring-key", "Scoring Receiver")
    prediction["version"] = league
    model_context = pd.DataFrame(
        [
            _scoring_match_context(
                "scoring-key",
                source="model_inputs_projection_context",
                total_points=100.0,
                projection_ppg=6.0,
                pass_points=1.0,
                rush_points=19.0,
                receiving_points=80.0,
                team_qb1_points=250.0,
                projection_std_points=5.0,
            )
        ]
    )
    v2_context = pd.DataFrame(
        [
            _scoring_match_context(
                "scoring-key",
                source="v2_player_season_features_scoring_context",
                total_points=204.0,
                projection_ppg=12.0,
                pass_points=-20.4,
                rush_points=102.0,
                receiving_points=122.4,
                team_qb1_points=374.0,
                projection_std_points=25.5,
            )
        ]
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league(league)
        attached = weekly_builder.attach_current_context_by_player_key(
            pd.DataFrame([prediction]),
            model_context,
            v2_context,
        ).iloc[0]
    finally:
        weekly_builder.set_active_league(original_league)

    assert attached["current_avg_proj_points"] == pytest.approx(204.0)
    assert attached["avg_proj_pass_points"] == pytest.approx(-20.4)
    assert attached["avg_proj_rush_points"] == pytest.approx(102.0)
    assert attached["avg_proj_rec_points"] == pytest.approx(122.4)
    assert attached["std_proj_points"] == pytest.approx(25.5)


@pytest.mark.parametrize("league", ["dk", "beta"])
def test_non_nffc_current_join_preserves_model_input_scoring_context(league):
    prediction = _current_prediction("scoring-key", "Scoring Receiver")
    prediction["version"] = league
    model_context = pd.DataFrame(
        [
            _scoring_match_context(
                "scoring-key",
                source="model_inputs_projection_context",
                total_points=100.0,
                projection_ppg=6.0,
                pass_points=1.0,
                rush_points=19.0,
                receiving_points=80.0,
                team_qb1_points=250.0,
                projection_std_points=5.0,
            )
        ]
    )
    v2_context = pd.DataFrame(
        [
            _scoring_match_context(
                "scoring-key",
                source="v2_player_season_features_scoring_context",
                total_points=204.0,
                projection_ppg=12.0,
                pass_points=20.4,
                rush_points=61.2,
                receiving_points=122.4,
                team_qb1_points=374.0,
                projection_std_points=25.5,
            )
        ]
    )

    original_league = weekly_builder.LEAGUE
    try:
        weekly_builder.set_active_league(league)
        attached = weekly_builder.attach_current_context_by_player_key(
            pd.DataFrame([prediction]),
            model_context,
            v2_context,
            scoring_matched_context=False,
        ).iloc[0]
    finally:
        weekly_builder.set_active_league(original_league)

    assert attached["current_avg_proj_points"] == pytest.approx(100.0)
    assert attached["avg_proj_points"] == pytest.approx(100.0)
    assert attached["avg_proj_pass_points"] == pytest.approx(1.0)
    assert attached["avg_proj_rush_points"] == pytest.approx(19.0)
    assert attached["avg_proj_rec_points"] == pytest.approx(80.0)
    assert attached["qb_avg_proj_pass_points"] == pytest.approx(250.0)
    assert attached["std_proj_points"] == pytest.approx(5.0)
    assert attached["match_projection_ppg_scaled"] == pytest.approx(0.6)


def test_current_context_join_is_key_first_and_uses_v2_fallback():
    predictions = pd.DataFrame(
        [
            _current_prediction(
                "deebo-key",
                "Deebo Samuel Sr.",
            ),
            _current_prediction(
                "new-v2-key",
                "New V2 Display Name",
            ),
        ]
    )
    # The model-input display name intentionally differs from production. It is
    # absent from the join columns because identity was already resolved.
    model_context = pd.DataFrame(
        [
            {
                **_current_context(
                    "deebo-key",
                    source="model_inputs_projection_context",
                ),
                "player": "Deebo Samuel",
            }
        ]
    )
    fallback_context = pd.DataFrame(
        [
            _current_context(
                "deebo-key",
                source="v2_player_season_features_fallback",
            ),
            _current_context(
                "new-v2-key",
                source="v2_player_season_features_fallback",
                team="NEW",
            ),
        ]
    )

    attached = weekly_builder.attach_current_context_by_player_key(
        predictions,
        model_context,
        fallback_context,
        scoring_matched_context=False,
    ).set_index("player_key")

    assert attached.loc["deebo-key", "player"] == "Deebo Samuel Sr."
    assert (
        attached.loc["deebo-key", "current_context_source"]
        == "model_inputs_projection_context"
    )
    assert attached.loc["deebo-key", "current_context_fallback_fields"] == ""
    assert (
        attached.loc["new-v2-key", "current_context_source"]
        == "v2_player_season_features_fallback"
    )
    assert attached.loc["new-v2-key", "team"] == "NEW"
    assert "current_avg_proj_points" in attached.loc[
        "new-v2-key",
        "current_context_fallback_fields",
    ]
    assert attached["current_context_missing_fields"].eq("").all()


def test_canonical_adp_team_fills_only_unassigned_current_team():
    context = pd.DataFrame(
        {
            "team": [pd.NA, "FA", "KC"],
            "published_team": ["NYG", pd.NA, "BUF"],
        }
    )

    filled = weekly_builder.fill_current_team_from_published_adp(
        context,
        published_team_column="published_team",
        primary_source="v2_player_season_features",
    )

    assert filled["team"].tolist() == ["NYG", "FA", "KC"]
    assert filled["current_team_source"].tolist() == [
        "canonical_avg_adps",
        "unassigned",
        "v2_player_season_features",
    ]


def test_recommendation_row_without_required_keyed_context_fails():
    predictions = pd.DataFrame(
        [_current_prediction("missing-key", "Missing Context", ppg=10.0)]
    )
    empty_context = pd.DataFrame(columns=["player_key", "pos", "year"])

    with pytest.raises(
        ValueError,
        match=(
            "Recommendation-eligible production rows lack required "
            "key-first"
        ),
    ):
        weekly_builder.attach_current_context_by_player_key(
            predictions,
            empty_context,
            empty_context,
        )


def test_existing_production_player_key_is_preserved_without_name_resolution(
    tmp_path,
):
    identity_db = tmp_path / "identity.sqlite3"
    with sqlite3.connect(identity_db) as connection:
        connection.execute(
            "CREATE TABLE player_identity "
            "(player_key TEXT, position TEXT)"
        )
        connection.execute(
            "INSERT INTO player_identity VALUES ('canonical-key', 'WR')"
        )

    frame = pd.DataFrame(
        [
            {
                "player_key": "canonical-key",
                "player": "Provider Display Variant",
                "pos": "WR",
                "year": 2026,
            }
        ]
    )
    validated = weekly_builder.validate_existing_v2_player_keys(
        frame,
        identity_db,
    )

    assert validated.loc[0, "player_key"] == "canonical-key"
    assert (
        validated.loc[0, "player_key_match_method"]
        == "production_handoff_player_key"
    )


def test_selected_universe_team_alias_rooms_are_canonical_and_idempotent():
    rows = [
        ("la-qb-1", "LA QB1", "QB", "LA", 300.0, 280.0, 0.0),
        ("la-qb-2", "LA QB2", "QB", "LAR", 200.0, 180.0, 0.0),
        ("ari-qb-1", "ARI QB1", "QB", "ARZ", 250.0, 225.0, 0.0),
        ("ari-qb-2", "ARI QB2", "QB", "ARI", 150.0, 125.0, 0.0),
        ("la-wr", "LA Receiver", "WR", "LA", 100.0, 0.0, 60.0),
        ("la-te", "LAR Tight End", "TE", "LAR", 80.0, 0.0, 40.0),
        ("ari-wr", "ARZ Receiver", "WR", "ARZ", 70.0, 0.0, 30.0),
        ("ari-te", "ARI Tight End", "TE", "ARI", 90.0, 0.0, 70.0),
        ("fa-qb", "Free Agent QB", "QB", "FA", 175.0, 150.0, 0.0),
        ("fa-wr", "Free Agent WR", "WR", "FA", 50.0, 0.0, 50.0),
    ]
    player_map = pd.DataFrame(
        [
            {
                "player_key": player_key,
                "player": player,
                "pos": pos,
                "team": team,
                "year": 2026,
                "current_avg_proj_points": total_points,
                "avg_proj_points": total_points,
                "avg_pick": float(index + 1),
                "year_exp": 2.0,
                "avg_proj_pass_points": pass_points,
                "avg_proj_rush_points": 0.0,
                "avg_proj_rec_points": rec_points,
                "qb_avg_proj_pass_points": 0.0,
                "std_proj_points": 5.0,
                "std_pos_rank": 1.0,
            }
            for index, (
                player_key,
                player,
                pos,
                team,
                total_points,
                pass_points,
                rec_points,
            ) in enumerate(rows)
        ]
    )

    first = weekly_builder.recompute_selected_universe_match_features(
        player_map
    )
    second = weekly_builder.recompute_selected_universe_match_features(first)
    pd.testing.assert_frame_equal(first, second)

    keyed = first.set_index("player_key")
    assert keyed["team"].tolist() == player_map.set_index("player_key")[
        "team"
    ].tolist()
    assert keyed.loc["la-qb-1", "qb_team_rank"] == 1
    assert keyed.loc["la-qb-2", "qb_team_rank"] == 2
    assert keyed.loc["ari-qb-1", "qb_team_rank"] == 1
    assert keyed.loc["ari-qb-2", "qb_team_rank"] == 2
    assert keyed.loc["la-wr", "team_rec_share"] == pytest.approx(0.60)
    assert keyed.loc["la-te", "team_rec_share"] == pytest.approx(0.40)
    assert keyed.loc[
        "la-wr", "pass_catcher_room_concentration"
    ] == pytest.approx(0.52)
    assert keyed.loc["ari-wr", "team_rec_share"] == pytest.approx(0.30)
    assert keyed.loc["ari-te", "team_rec_share"] == pytest.approx(0.70)
    assert keyed.loc[
        "ari-te", "pass_catcher_room_concentration"
    ] == pytest.approx(0.58)
    assert keyed.loc["fa-qb", "qb_team_rank"] == -1
    assert keyed.loc["fa-qb", "qb_team_rank_bucket"] == "unknown"
    assert keyed.loc["fa-wr", "team_rec_share"] == 0
    assert keyed.loc["fa-wr", "pass_catcher_room_concentration"] == 0
