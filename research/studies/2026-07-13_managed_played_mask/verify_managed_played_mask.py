import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
GITHUB_ROOT = REPO_ROOT.parent
APP_DIR = GITHUB_ROOT / 'Fantasy_Football_App' / 'app'
APP_DB = APP_DIR / 'Simulation.sqlite3'
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT / 'Scripts'))
sys.path.insert(0, str(GITHUB_ROOT / 'ff'))

from zSim_Helper import FootballSimulation  # noqa: E402
import config as model_config  # noqa: E402
from Modeling import s4_Best_Ball_Weekly as weekly_builder  # noqa: E402


LINEUP = {'QB': 0, 'RB': 1, 'WR': 0, 'TE': 0, 'FLEX': 0}
WAIVERS = {'QB': 0, 'RB': 5, 'WR': 0, 'TE': 0}


def verify_builder_contract():
    projection = pd.DataFrame([{
        'player': 'Played Zero',
        'pos': 'RB',
        'team': 'TST',
        'season': 2024,
        'avg_proj_points': 120.0,
        'preseason_proj_ppg': 8.0,
        'validation_pred_fp_per_game': 8.0,
        'historical_pred_fp_per_game': 8.0,
        'historical_projection_source': 'fixture',
        'validation_ensemble_sources': 1,
        'avg_pick': 100.0,
        'year_exp': 2,
        'year_exp_bucket': 2,
        'exp_bucket': 'young',
        'qb_team_rank': 0,
        'qb_team_rank_bucket': 'non_qb',
        'projection_rank_pct': 0.5,
        'projection_decile': 5,
        'projection_tier': 5,
    }])
    weekly = pd.DataFrame({
        'player': ['Played Zero'] * 3,
        'pos': ['RB'] * 3,
        'season': [2024] * 3,
        'week': [1, 2, 3],
        'fantasy_pts': [0.0, -1.0, 10.0],
    })
    old_league = weekly_builder.LEAGUE
    weekly_builder.LEAGUE = 'beta'
    try:
        templates = weekly_builder.build_weekly_templates(projection, weekly)
    finally:
        weekly_builder.LEAGUE = old_league

    row = templates.iloc[0]
    played = row[[f'played_week_{week}' for week in range(1, 17)]].to_numpy(
        dtype=int,
    )
    np.testing.assert_array_equal(
        played,
        np.array([1, 1, 1] + [0] * 13),
    )
    assert int(row.active_games) == 3
    assert int(row.played_games) == 3
    assert float(row.week_1) == 0
    assert float(row.week_2) < 0

    qb_projection = projection.copy()
    qb_projection.loc[:, 'player'] = 'Short QB Appearance'
    qb_projection.loc[:, 'pos'] = 'QB'
    qb_projection.loc[:, 'qb_team_rank'] = 2
    qb_projection.loc[:, 'qb_team_rank_bucket'] = 'qb2'
    qb_weekly = pd.DataFrame({
        'player': ['Short QB Appearance'] * 3,
        'pos': ['QB'] * 3,
        'season': [2024] * 3,
        'week': [1, 2, 3],
        'fantasy_pts': [np.nan, np.nan, 10.0],
        'managed_fantasy_pts': [4.0, -1.0, 10.0],
        'played_week': [True, True, True],
    })
    weekly_builder.LEAGUE = 'beta'
    try:
        qb_templates = weekly_builder.build_weekly_templates(qb_projection, qb_weekly)
    finally:
        weekly_builder.LEAGUE = old_league
    qb_row = qb_templates.iloc[0]
    qb_played = qb_row[[f'played_week_{week}' for week in range(1, 17)]].to_numpy(
        dtype=int,
    )
    np.testing.assert_array_equal(qb_played, np.array([1, 1, 1] + [0] * 13))
    assert int(qb_row.active_games) == 1
    assert int(qb_row.played_games) == 3
    assert float(qb_row.week_1) == 0
    assert float(qb_row.week_2) == 0
    assert float(qb_row.profile_total) == 1
    assert np.isclose(float(qb_row.managed_week_1), 0.4)
    assert np.isclose(float(qb_row.managed_week_2), -0.1)
    assert np.isclose(float(qb_row.managed_profile_total), 1.3)
    qb_audit = weekly_builder.build_template_join_audit(qb_templates).iloc[0]
    assert int(qb_audit.played_only_games) == 2
    assert not bool(qb_audit.played_mask_mismatch)
    return {
        'active_games': 3,
        'played_games': 3,
        'played_mask': played.tolist(),
        'short_qb_active_games': 1,
        'short_qb_played_games': 3,
    }


def verify_scalar_and_multi_lineups():
    scores = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
    decisions = np.full_like(scores, 10.0)
    played = np.array([[1, 1, 0]], dtype=np.int8)
    points, details = FootballSimulation.managed_lineup_weekly_scores(
        scores,
        np.array(['RB'], dtype=object),
        decisions,
        np.array(['Test RB'], dtype=object),
        lineup_require=LINEUP,
        waiver_baselines=WAIVERS,
        return_details=True,
        played_mask=played,
    )
    np.testing.assert_array_equal(points, np.array([0, -1, 5], dtype=np.float32))
    assert details.selected_players.iloc[0] == ['Test RB']
    assert details.selected_players.iloc[1] == ['Test RB']
    assert details.selected_players.iloc[2][0].startswith('WW_RB_')

    qb_points = FootballSimulation.managed_lineup_weekly_scores(
        np.array([[0.0]], dtype=np.float32),
        np.array(['QB'], dtype=object),
        np.array([[10.0]], dtype=np.float32),
        np.array(['Short QB Appearance'], dtype=object),
        lineup_require={'QB': 1, 'RB': 0, 'WR': 0, 'TE': 0, 'FLEX': 0},
        waiver_baselines={'QB': 5, 'RB': 0, 'WR': 0, 'TE': 0},
        played_mask=np.array([[1]], dtype=np.int8),
    )
    np.testing.assert_array_equal(qb_points, np.array([0], dtype=np.float32))

    context_scores = np.array([
        [[0.0, -1.0, 0.0]],
        [[0.0, 0.0, -2.0]],
    ], dtype=np.float32)
    context_decisions = np.full_like(context_scores, 10.0)
    context_played = np.array([
        [[1, 1, 0]],
        [[0, 1, 1]],
    ], dtype=np.int8)
    season_scores, starts = FootballSimulation.managed_lineup_multi_context_scores(
        context_scores,
        np.array(['RB'], dtype=object),
        context_decisions,
        np.array(['Test RB'], dtype=object),
        lineup_require=LINEUP,
        waiver_baselines=WAIVERS,
        tracked_player='Test RB',
        played_mask=context_played,
    )
    np.testing.assert_array_equal(season_scores, np.array([4, 3], dtype=np.float32))
    np.testing.assert_array_equal(starts, np.array([2, 2], dtype=np.int32))
    for context_idx in range(2):
        scalar = FootballSimulation.managed_lineup_weekly_scores(
            context_scores[context_idx],
            np.array(['RB'], dtype=object),
            context_decisions[context_idx],
            np.array(['Test RB'], dtype=object),
            lineup_require=LINEUP,
            waiver_baselines=WAIVERS,
            played_mask=context_played[context_idx],
        )
        assert float(scalar.sum()) == float(season_scores[context_idx])

    try:
        FootballSimulation.managed_lineup_multi_context_scores(
            context_scores,
            np.array(['RB'], dtype=object),
            context_decisions,
            np.array(['Test RB'], dtype=object),
            lineup_require=LINEUP,
            waiver_baselines=WAIVERS,
            played_mask=np.ones((2, 1, 2), dtype=np.int8),
        )
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched played-mask shape was not rejected.')

    return {'scalar_weekly_points': points.tolist(), 'multi_season_points': season_scores.tolist()}


def verify_learning_and_marginals():
    learning_scores = np.array([[10.0, 0.0, 5.0]], dtype=np.float32)
    missed = FootballSimulation.build_managed_decision_scores(
        learning_scores,
        preseason_ppg=np.array([8.0]),
        learn_weeks=0,
        max_learn_weight=1.0,
        played_mask=np.array([[1, 0, 1]], dtype=np.int8),
    )
    played_zero = FootballSimulation.build_managed_decision_scores(
        learning_scores,
        preseason_ppg=np.array([8.0]),
        learn_weeks=0,
        max_learn_weight=1.0,
        played_mask=np.array([[1, 1, 1]], dtype=np.int8),
    )
    assert float(missed[0, 2]) == 10.0
    assert float(played_zero[0, 2]) == 5.0

    scores = np.array([
        [4.0, 4.0, 4.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float32)
    decisions = np.array([
        [5.0, 5.0, 5.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float32)
    played = np.array([
        [1, 1, 1],
        [1, 1, 0],
    ], dtype=np.int8)
    values = FootballSimulation.managed_marginal_values_batch(
        scores,
        np.array(['RB', 'RB'], dtype=object),
        decisions,
        np.array(['Base', 'Candidate'], dtype=object),
        [['Base'], []],
        lineup_require=LINEUP,
        waiver_baselines={**WAIVERS, 'RB': 2},
        bench_upside_weight=0,
        played_mask=played,
    )
    np.testing.assert_array_equal(
        values[:, 1],
        np.array([-9, -5], dtype=np.float32),
    )
    return {
        'missed_week_learned_ppg': float(missed[0, 2]),
        'played_zero_learned_ppg': float(played_zero[0, 2]),
        'candidate_marginals': values[:, 1].tolist(),
    }


def make_loader_db(mode):
    conn = sqlite3.connect(':memory:')
    template = {
        'template_id': 1,
        'active_ppg_resid': 0.0,
        'week_1': 0.0,
        'week_2': 1.0,
    }
    if mode in ('complete', 'partial', 'managed', 'managed_partial'):
        template['played_week_1'] = 1
    if mode in ('complete', 'managed', 'managed_partial'):
        template['played_week_2'] = 0
    if mode in ('managed', 'managed_partial'):
        template['managed_week_1'] = 3.0
    if mode == 'managed':
        template['managed_week_2'] = 4.0
    pd.DataFrame([template]).to_sql(
        'Best_Ball_Weekly_Templates', conn, index=False,
    )
    pd.DataFrame([{
        'template_pool_key': 'pool',
        'template_id': 1,
        'template_sample_prob': 1.0,
        'match_rank': 1,
    }]).to_sql('Best_Ball_Weekly_Template_Pools', conn, index=False)
    pd.DataFrame([{
        'player': 'Loader Player',
        'template_pool_key': 'pool',
        'year': 2026,
        'version': 'beta',
        'dataset': 'final_ensemble',
    }]).to_sql('Best_Ball_Weekly_Player_Map', conn, index=False)
    return conn


def verify_loader_compatibility():
    complete = make_loader_db('complete')
    try:
        cache = FootballSimulation.read_weekly_template_profile_cache(
            complete, 2026, 'beta', 'final_ensemble',
        )
        np.testing.assert_array_equal(
            cache[-1]['Loader Player'],
            np.array([[1, 0]], dtype=np.int8),
        )
    finally:
        complete.close()

    managed = make_loader_db('managed')
    try:
        cache = FootballSimulation.read_weekly_template_profile_cache(
            managed, 2026, 'beta', 'final_ensemble',
        )
        np.testing.assert_array_equal(
            cache[1]['Loader Player'],
            np.array([[3, 4]], dtype=np.float32),
        )
    finally:
        managed.close()

    legacy = make_loader_db('legacy')
    try:
        cache = FootballSimulation.read_weekly_template_profile_cache(
            legacy, 2026, 'beta', 'final_ensemble',
        )
        np.testing.assert_array_equal(
            cache[-1]['Loader Player'],
            np.array([[-1, -1]], dtype=np.int8),
        )
    finally:
        legacy.close()

    partial = make_loader_db('partial')
    try:
        try:
            FootballSimulation.read_weekly_template_profile_cache(
                partial, 2026, 'beta', 'final_ensemble',
            )
        except ValueError as exc:
            assert 'partial played-week schema' in str(exc)
        else:
            raise AssertionError('Partial played-week schema was not rejected.')
    finally:
        partial.close()

    managed_partial = make_loader_db('managed_partial')
    try:
        try:
            FootballSimulation.read_weekly_template_profile_cache(
                managed_partial, 2026, 'beta', 'final_ensemble',
            )
        except ValueError as exc:
            assert 'partial managed-week schema' in str(exc)
        else:
            raise AssertionError('Partial managed-week schema was not rejected.')
    finally:
        managed_partial.close()
    return {
        'complete_mask': [1, 0],
        'legacy_mask': [-1, -1],
        'managed_profile': [3, 4],
    }


def verify_sampling_pairing():
    sim = FootballSimulation.__new__(FootballSimulation)
    sim.weekly_template_week_cols = ['week_1', 'week_2']
    sim.weekly_template_profiles = {
        'Pair Player': np.array([[0, 1], [2, 0]], dtype=np.float32),
    }
    sim.weekly_template_played_masks = {
        'Pair Player': np.array([[1, 1], [1, 0]], dtype=np.int8),
    }
    sim.weekly_template_cum_probs = {
        'Pair Player': np.array([0.5, 1.0]),
    }
    sim.weekly_template_centered_active_ppg_resids = {
        'Pair Player': np.array([0, 0], dtype=np.float32),
    }
    sim.weekly_template_active_ppg_resid_sds = {'Pair Player': 0.0}
    predictions = pd.DataFrame([{
        'player': 'Pair Player',
        'pos': 'RB',
        'value_1': 10.0,
    }])
    scores, masks = sim.sample_seeded_template_weekly_contexts(
        predictions,
        range(50),
        num_weeks=2,
        return_played_masks=True,
    )
    observed = {
        (
            tuple(scores[idx, 0].tolist()),
            tuple(masks[idx, 0].tolist()),
        )
        for idx in range(len(scores))
    }
    expected = {
        ((0.0, 10.0), (1, 1)),
        ((20.0, 0.0), (1, 0)),
    }
    assert observed == expected
    return {'paired_template_outcomes': len(observed)}


def verify_source_participation():
    old_builder_league = weekly_builder.LEAGUE
    old_config_league = model_config.LEAGUE
    weekly_builder.LEAGUE = 'beta'
    model_config.LEAGUE = 'beta'
    try:
        max_template_season = min(
            weekly_builder.YEAR - 1,
            weekly_builder.get_daily_max_template_season(),
        )
        projection_keys = weekly_builder.load_historical_projection_context(
            max_template_season,
        )[['player', 'pos', 'season']].drop_duplicates()
        weekly = weekly_builder.load_weekly_points(max_template_season)
    finally:
        weekly_builder.LEAGUE = old_builder_league
        model_config.LEAGUE = old_config_league

    source = projection_keys.merge(
        weekly,
        on=['player', 'pos', 'season'],
        how='inner',
        validate='one_to_many',
    )
    played = source[source.played_week.eq(True)].copy()
    played_only = played.fantasy_pts.isna()
    non_qb_played_only = played_only & played.pos.ne('QB')
    if non_qb_played_only.any():
        raise AssertionError('Only QBs may have participation-only source weeks.')
    if (played.fantasy_pts.notna().sum() + played_only.sum()) != len(played):
        raise AssertionError('Played source rows were not fully classified.')
    if played.managed_fantasy_pts.isna().any():
        raise AssertionError('Played source rows are missing managed fantasy scores.')
    short_qb = played[played_only]
    return {
        'played_weeks': int(len(played)),
        'workload_qualified_weeks': int(played.fantasy_pts.notna().sum()),
        'exact_zero_scores': int(
            np.isclose(played.managed_fantasy_pts, 0).sum()
        ),
        'negative_scores': int(played.managed_fantasy_pts.lt(0).sum()),
        'qb_short_appearance_weeks': int(played_only.sum()),
        'qb_short_nonzero_weeks': int(
            (~np.isclose(short_qb.managed_fantasy_pts, 0)).sum()
        ),
        'qb_short_total_points': float(short_qb.managed_fantasy_pts.sum()),
    }


def verify_rebuilt_database():
    conn = sqlite3.connect(APP_DB)
    try:
        table_cols = {
            row[1]
            for row in conn.execute(
                'PRAGMA table_info(Best_Ball_Weekly_Templates)'
            )
        }
        played_cols = [f'played_week_{week}' for week in range(1, 17)]
        managed_cols = [f'managed_week_{week}' for week in range(1, 17)]
        missing = sorted(set(played_cols + managed_cols) - table_cols)
        if missing:
            raise AssertionError(f'Rebuilt app DB is missing weekly columns: {missing}')
        output = {}
        for league in ('beta', 'dk'):
            frame = pd.read_sql_query(
                'SELECT active_games, played_games, pos, '
                + ', '.join(managed_cols + played_cols)
                + ' FROM Best_Ball_Weekly_Templates WHERE league = ?',
                conn,
                params=(league,),
            )
            if len(frame) == 0:
                raise AssertionError(f'Rebuilt app DB has no {league} weekly templates.')
            played = frame[played_cols].to_numpy(dtype=float)
            if not np.isfinite(played).all() or not np.isin(played, [0, 1]).all():
                raise AssertionError(
                    f'Rebuilt {league} played masks are not complete 0/1 values.'
                )
            np.testing.assert_array_equal(
                played.sum(axis=1).astype(int),
                frame.played_games.to_numpy(dtype=int),
            )
            played_only = (
                frame.played_games.to_numpy(dtype=int)
                - frame.active_games.to_numpy(dtype=int)
            )
            if np.any(played_only < 0):
                raise AssertionError('Played-game counts cannot be below active-game counts.')
            if np.any(played_only[frame.pos.ne('QB').to_numpy()] != 0):
                raise AssertionError('Non-QB active and played counts must match.')
            qb_played_only = int(played_only[frame.pos.eq('QB').to_numpy()].sum())
            if qb_played_only == 0:
                raise AssertionError(f'Rebuilt {league} masks lost short QB appearances.')
            scores = frame[managed_cols].to_numpy(float)
            if not np.isfinite(scores).all():
                raise AssertionError(f'Rebuilt {league} managed profiles are incomplete.')
            played_zero_profile = int(((played == 1) & np.isclose(scores, 0)).sum())
            played_negative = int(((played == 1) & (scores < 0)).sum())
            if played_zero_profile == 0 or played_negative == 0:
                raise AssertionError(
                    f'Rebuilt {league} masks did not preserve played downside outcomes.'
                )
            output[league] = {
                'templates': int(len(frame)),
                'played_zero_managed_profile_weeks': played_zero_profile,
                'played_negative_managed_profile_weeks': played_negative,
                'qb_short_appearance_weeks': qb_played_only,
            }
        return output
    finally:
        conn.close()


def main():
    output = {
        'builder': verify_builder_contract(),
        'lineups': verify_scalar_and_multi_lineups(),
        'learning_and_marginals': verify_learning_and_marginals(),
        'loader': verify_loader_compatibility(),
        'sampling': verify_sampling_pairing(),
        'source': verify_source_participation(),
        'database': verify_rebuilt_database(),
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
