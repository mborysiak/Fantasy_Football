from argparse import ArgumentParser
from pathlib import Path
import sqlite3
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = ROOT.parent / 'Fantasy_Football_App'
sys.path.insert(0, str(APP_ROOT / 'app'))

from zSim_Helper import (  # noqa: E402
    DEFAULT_TARGET_PILOT_DISCOVERIES,
    DEFAULT_TARGET_PILOT_TRIALS,
    DEFAULT_TARGET_CONFIRM_CANDIDATES,
    DEFAULT_TARGET_MARKET_CONFIRM_ANCHORS,
    DEFAULT_TARGET_CONFIRM_TRIALS,
    DEFAULT_TARGET_SCREEN_CANDIDATES,
    DEFAULT_TARGET_SCREEN_TRIALS,
    FootballSimulation,
    MANAGED_POS_MAX,
)


FIXED_PLAYERS = [
    'Jayden Daniels',
    'Jahmyr Gibbs',
    'Chase Brown',
    'Bhayshul Tuten',
]
FIXED_SALARIES = [15, 108, 34, 11]
LINEUP_REQUIRE = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
WAIVER_BASELINES = {'QB': 15.7, 'RB': 5.5, 'WR': 6.0, 'TE': 6.5}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--market-confirm-anchors',
        type=int,
        default=DEFAULT_TARGET_MARKET_CONFIRM_ANCHORS,
    )
    args = parser.parse_args()
    market_confirm_anchor_count = max(0, args.market_confirm_anchors)

    database_path = APP_ROOT / 'app' / 'Simulation.sqlite3'
    conn = sqlite3.connect(database_path)
    try:
        keepers = pd.read_sql_query(
            """SELECT player, keeper_salary
                 FROM League_Keepers
                WHERE year=2026 AND league='beta'""",
            conn,
        )
        simulation = FootballSimulation(
            conn,
            2026,
            LINEUP_REQUIRE,
            298,
            'final_ensemble',
            'beta',
            sal_pred_actual='pred',
        )

        fixed_keeper_players = {'Chase Brown', 'Bhayshul Tuten'}
        to_drop = keepers.loc[
            ~keepers.player.isin(fixed_keeper_players),
            'player',
        ].tolist()
        keeper_spend = float(keepers.keeper_salary.sum())
        nonkeeper_fixed_spend = 15 + 108
        remaining_market_budget = 12 * 298 - keeper_spend - nonkeeper_fixed_spend
        remaining_market_slots = 12 * 13 - len(keepers) - 2

        start = time.perf_counter()
        results = simulation.run_sim_parallel(
            {'players': FIXED_PLAYERS, 'salaries': FIXED_SALARIES},
            to_drop,
            320,
            max_workers=8,
            random_seed=20260717,
            require_top_n=12,
            num_avg_pts=5,
            next_year_frac=0.0,
            enforce_top_n=True,
            scoring_mode='managed',
            roster_size=13,
            lineup_require=LINEUP_REQUIRE,
            pos_min_counts={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1},
            pos_max_counts=MANAGED_POS_MAX,
            waiver_baselines=WAIVER_BASELINES,
            bench_upside_weight=0.25,
            remaining_market_budget=remaining_market_budget,
            remaining_market_slots=remaining_market_slots,
            use_selection_premium=True,
            target_screen_size=DEFAULT_TARGET_SCREEN_CANDIDATES,
            target_screen_trials=DEFAULT_TARGET_SCREEN_TRIALS,
            target_pilot_trials=DEFAULT_TARGET_PILOT_TRIALS,
            target_pilot_discoveries=DEFAULT_TARGET_PILOT_DISCOVERIES,
            target_confirm_size=DEFAULT_TARGET_CONFIRM_CANDIDATES,
            target_market_confirm_anchors=market_confirm_anchor_count,
            target_confirm_trials=DEFAULT_TARGET_CONFIRM_TRIALS,
        )
        elapsed = time.perf_counter() - start
    finally:
        conn.close()

    audit_columns = [
        'player',
        'TargetOrder',
        'EvidenceTier',
        'RankFamily',
        'RecommendationStatus',
        'TargetRankScore',
        'ForcedActionScore',
        'EvidenceStatus',
        'ConfirmationRoute',
        'MarketConfirmationAnchor',
        'SelectionCounts',
        'ContributionN',
        'PrelimGain',
        'PrelimN',
        'ConfirmGain',
        'ConfirmN',
        'ConfirmAttempts',
        'PosteriorGain',
        'PosteriorSE',
        'IndependentPosteriorSE',
        'StageDisagreementTau',
        'PosteriorProbPositive',
        'ReplicationZ',
        'BuyPosMix',
        'BuyWRTEBudget',
        'WRTEBudgetDelta',
        'BuyWRTECore',
    ]
    print(results[[
        column for column in audit_columns if column in results
    ]].head(20).to_string(index=False))
    key_players = [
        'Derrick Henry',
        'Saquon Barkley',
        'Jonathan Taylor',
        'Kyren Williams',
        'Ashton Jeanty',
        "Ja'Marr Chase",
        'Chris Olave',
        'Ladd Mcconkey',
    ]
    print()
    print('Key-player audit:')
    print(results.loc[
        results.player.isin(key_players),
        [column for column in audit_columns if column in results],
    ].to_string(index=False))
    print()
    print(f'Elapsed seconds: {elapsed:.1f}')
    print(
        'Workers: '
        f'organic={simulation.parallel_workers_used}, '
        f'later={simulation.target_stage_workers_used}'
    )
    print(f'Confirmed candidates: {simulation.target_confirm_candidates}')
    print(
        'Evidence-priority confirmations: '
        f'{simulation.target_evidence_confirm_candidates}'
    )
    print(
        'Market confirmation anchors: '
        f'{simulation.target_market_confirm_anchors}'
    )
    print(f'Pilot discoveries: {simulation.target_pilot_discoveries}')
    print(f'Preliminary candidates: {simulation.target_screen_candidates}')
    assert simulation.parallel_workers_used == 8
    assert simulation.target_organic_blocks == [40] * 8
    assert simulation.target_stage_workers_used == 8
    assert simulation.target_screen_trials == DEFAULT_TARGET_SCREEN_TRIALS
    assert simulation.target_pilot_trials == DEFAULT_TARGET_PILOT_TRIALS
    assert simulation.target_confirm_trials == DEFAULT_TARGET_CONFIRM_TRIALS
    assert simulation.target_confirm_candidates == (
        simulation.target_confirmation_cohort(
            simulation.target_evidence_confirm_candidates,
            simulation.target_market_confirm_anchors,
            evidence_limit=DEFAULT_TARGET_CONFIRM_CANDIDATES,
            market_anchor_limit=market_confirm_anchor_count,
        )
    )
    assert set(simulation.target_evidence_confirm_candidates).issubset(
        simulation.target_confirm_candidates
    )
    assert set(simulation.target_market_confirm_anchors).issubset(
        simulation.target_confirm_candidates
    )
    assert len(simulation.target_confirm_candidates) <= (
        DEFAULT_TARGET_CONFIRM_CANDIDATES
        + market_confirm_anchor_count
    )
    market_anchor_rows = results[results.player.isin(
        simulation.target_market_confirm_anchors
    )]
    assert len(market_anchor_rows) == len(
        simulation.target_market_confirm_anchors
    )
    assert market_anchor_rows.MarketConfirmationAnchor.all()
    assert market_anchor_rows.ConfirmationRoute.str.contains(
        'market anchor',
        case=False,
    ).all()
    assert (
        market_anchor_rows.ConfirmAttempts == DEFAULT_TARGET_CONFIRM_TRIALS
    ).all()
    assert len(simulation.target_pilot_discoveries) <= (
        DEFAULT_TARGET_PILOT_DISCOVERIES
    )
    assert set(simulation.target_pilot_discoveries).issubset(
        simulation.target_screen_candidates
    )
    assert len(simulation.target_screen_candidates) == (
        DEFAULT_TARGET_SCREEN_CANDIDATES
    )
    assert (
        results.PrelimN.fillna(0) <= DEFAULT_TARGET_SCREEN_TRIALS
    ).all()
    assert (
        results.ConfirmN.fillna(0) <= DEFAULT_TARGET_CONFIRM_TRIALS
    ).all()
    assert results.BuyPosMix.notna().any()
    assert results.RankFamilyOrder.is_monotonic_increasing
    forced = results[results.RankFamily == 'Forced']
    assert forced.PrelimUsable.all()
    assert (forced.TargetRankScore == forced.ForcedActionScore).all()
    assert forced.TargetRankScore.is_monotonic_decreasing
    confirmed = forced[forced.ConfirmUsable]
    assert set(confirmed.EvidenceTier) == {'Confirmed'}
    assert (
        confirmed.PosteriorSE >= confirmed.IndependentPosteriorSE
    ).all()
    preliminary = forced[~forced.ConfirmUsable]
    assert set(preliminary.EvidenceTier) <= {'Preliminary'}
    print('Staged-evidence edge smoke checks passed.')


if __name__ == '__main__':
    main()
