from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = ROOT.parent / 'Fantasy_Football_App'
sys.path.insert(0, str(APP_ROOT / 'app'))

from zSim_Helper import (  # noqa: E402
    FootballSimulation,
    TARGET_ACTION_LCB_Z,
)


def verify_continuous_ranking():
    players = [f'P{idx}' for idx in range(7)]
    base = pd.DataFrame({
        'player': players,
        'SelectionCounts': [20.0] * 7,
        'ContributionN': [20] * 7,
        'ContributionCoverage': [100.0] * 7,
        'ExpectedRosterGain': [999.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    })
    prelim = pd.DataFrame({
        'player': players[:6],
        'PrelimGain': [3.0] * 6,
        'PrelimSE': [1.0] * 6,
        'PrelimBlocks': [8] * 6,
        'PrelimN': [40] * 6,
        'PrelimAttempts': [40] * 6,
        'PrelimPairedRate': [100.0] * 6,
    })
    confirm = pd.DataFrame({
        'player': ['P0', 'P1'],
        'ConfirmGain': [-0.1, 4.0],
        'ConfirmSE': [1.0, 1.0],
        'ConfirmBlocks': [8, 8],
        'ConfirmN': [60, 60],
        'ConfirmAttempts': [60, 60],
        'ConfirmPairedRate': [100.0, 100.0],
    })

    merged = FootballSimulation.merge_target_screen_results(
        base,
        prelim,
        confirmation_results=confirm,
        pilot_candidates=['P1'],
        evidence_confirmation_candidates=['P1'],
        market_confirmation_anchors=['P0', 'P1'],
    )
    ranked = FootballSimulation.finalize_target_ranking(merged)
    by_player = ranked.set_index('player')

    assert by_player.at['P1', 'EvidenceTier'] == 'Confirmed'
    assert by_player.at['P1', 'EvidenceStatus'] == 'Strong confirmation'
    assert by_player.at['P0', 'EvidenceTier'] == 'Confirmed'
    assert by_player.at['P0', 'EvidenceStatus'] == 'Negative confirmation'
    assert bool(by_player.at['P0', 'MarketConfirmationAnchor'])
    assert by_player.at['P0', 'ConfirmationRoute'] == 'Market anchor'
    assert by_player.at['P1', 'ConfirmationRoute'] == (
        'Evidence priority + market anchor'
    )
    assert set(by_player.loc[['P2', 'P3', 'P4', 'P5'], 'EvidenceTier']) == {
        'Preliminary'
    }
    assert set(by_player.loc[players[:6], 'RankFamily']) == {'Forced'}
    assert by_player.at['P6', 'RankFamily'] == 'Organic'
    assert ranked.iloc[0].player == 'P1'
    assert ranked.iloc[-1].player == 'P6'
    assert ranked.TargetOrder.tolist() == list(range(1, 8))
    expected_action_score = (
        by_player.at['P1', 'PosteriorGain']
        - TARGET_ACTION_LCB_Z * by_player.at['P1', 'PosteriorSE']
    )
    assert np.isclose(
        by_player.at['P1', 'ForcedActionScore'],
        expected_action_score,
    )
    assert by_player.at['P1', 'TargetRankScore'] == (
        by_player.at['P1', 'ForcedActionScore']
    )
    assert by_player.at['P0', 'TargetRankScore'] == (
        by_player.at['P0', 'ForcedActionScore']
    )
    assert by_player.at['P0', 'StageDisagreementTau'] > 0
    assert by_player.at['P0', 'PosteriorSE'] > (
        by_player.at['P0', 'IndependentPosteriorSE']
    )
    assert by_player.at['P1', 'StageDisagreementTau'] == 0
    assert np.isclose(
        by_player.at['P1', 'PosteriorSE'],
        by_player.at['P1', 'IndependentPosteriorSE'],
    )
    assert by_player.at['P6', 'TargetRankScore'] == 5.0

    without_pilot = FootballSimulation.merge_target_screen_results(
        base,
        prelim,
        confirmation_results=confirm,
    ).set_index('player')
    evidence_columns = [
        'PosteriorGain',
        'PosteriorSE',
        'PosteriorProbPositive',
        'ForcedActionScore',
        'TargetRankScore',
        'StageDisagreementTau',
    ]
    pd.testing.assert_frame_equal(
        by_player[evidence_columns].sort_index(),
        without_pilot[evidence_columns].sort_index(),
        check_exact=True,
    )


def verify_pilot_allocation():
    blocks = []
    scenario_values = (
        {'D1': 10.0, 'D2': 7.0, 'D3': 9.0, 'D4': 8.0, 'Incomplete': 100.0},
        {'D1': 8.0, 'D2': 9.0, 'D3': 7.0, 'D4': 6.0},
        {'D1': 9.0, 'D2': 8.0, 'D3': 8.0, 'D4': 7.0},
        {'D1': 9.0, 'D2': 8.0, 'D3': 8.0, 'D4': 7.0},
    )
    for block_index, values in enumerate(scenario_values):
        blocks.append({
            'block_index': block_index,
            'target_screen': {
                player: {'attempts': 1, 'contribution': [gain]}
                for player, gain in values.items()
            },
        })
    discoveries = FootballSimulation.target_pilot_discovery_shortlist(
        blocks,
        limit=4,
    )
    assert discoveries == ['D1', 'D2', 'D3', 'D4']
    assert 'Incomplete' not in discoveries

    heuristic = [f'H{idx}' for idx in range(1, 25)]
    cohort = FootballSimulation.target_preliminary_cohort(
        heuristic,
        discoveries,
        limit=24,
        pilot_slots=4,
    )
    assert cohort == heuristic[:20] + discoveries
    fallback = FootballSimulation.target_preliminary_cohort(
        heuristic,
        discoveries[:2],
        limit=24,
        pilot_slots=4,
    )
    assert fallback == heuristic[:20] + discoveries[:2] + heuristic[20:22]


def verify_confirmation_allocation():
    simulation = FootballSimulation.__new__(FootballSimulation)
    players = [
        'Incomplete',
        'Anchor A',
        'Anchor B',
        'Anchor C',
        'Anchor D',
        'Anchor E',
        'Evidence 1',
        'Evidence 2',
        'Evidence 3',
        'Evidence 4',
        'Evidence 5',
        'Evidence 6',
        'Evidence 7',
        'Evidence 8',
        'Evidence 9',
        'Evidence 10',
    ]
    simulation.player_data = pd.DataFrame({
        'player': players,
        'salary': [200.0, 120.0, 110.0, 100.0, 100.0, 90.0]
        + [80.0 - idx for idx in range(10)],
    })
    prelim = pd.DataFrame({
        'player': players,
        'PrelimGain': [1.0] * len(players),
        'PrelimSE': [1.0] * len(players),
        'PrelimBlocks': [8] * len(players),
        'PrelimN': [20] + [40] * (len(players) - 1),
        'PrelimAttempts': [40] * len(players),
        'PrelimPairedRate': [50.0] + [100.0] * (len(players) - 1),
    })

    market_anchors = simulation.target_market_confirmation_shortlist(
        prelim,
        limit=4,
    )
    assert market_anchors == [
        'Anchor A',
        'Anchor B',
        'Anchor C',
        'Anchor D',
    ]
    simulation.player_data = simulation.player_data.sample(
        frac=1.0,
        random_state=17,
    ).reset_index(drop=True)
    shuffled_anchors = simulation.target_market_confirmation_shortlist(
        prelim.sample(frac=1.0, random_state=18).reset_index(drop=True),
        limit=4,
    )
    assert shuffled_anchors == market_anchors

    evidence_order = [
        'Evidence 1',
        'Anchor B',
        'Evidence 2',
        'Evidence 3',
        'Evidence 4',
        'Evidence 5',
        'Evidence 6',
        'Evidence 7',
        'Evidence 8',
        'Evidence 9',
        'Evidence 10',
    ]
    cohort = FootballSimulation.target_confirmation_cohort(
        evidence_order,
        market_anchors,
        evidence_limit=10,
        market_anchor_limit=4,
    )
    assert cohort == evidence_order[:10] + [
        'Anchor A',
        'Anchor C',
        'Anchor D',
    ]
    assert len(cohort) == 13
    assert 'Incomplete' not in cohort
    assert FootballSimulation.target_confirmation_cohort(
        ['E1', 'E1', 'E2', 'E3'],
        ['M1', 'M1', 'E2', 'M2'],
        evidence_limit=3,
        market_anchor_limit=3,
    ) == ['E1', 'E2', 'E3', 'M1', 'M2']
    assert FootballSimulation.target_confirmation_cohort(
        evidence_order[:10],
        list(reversed(evidence_order[:4])),
        evidence_limit=10,
        market_anchor_limit=4,
    ) == evidence_order[:10]
    assert FootballSimulation.target_confirmation_cohort(
        ['E1'],
        ['M1'],
        evidence_limit=0,
        market_anchor_limit=4,
    ) == []

    tie_players = [
        'Zulu',
        'Alpha',
        'Hotel',
        'Golf',
        'Foxtrot',
        'Echo',
        'Delta',
        'Charlie',
        'Bravo',
        'Able',
        'Baker',
    ]
    tie_base = pd.DataFrame({
        'player': tie_players,
        'SelectionCounts': [20.0] * len(tie_players),
        'ContributionN': [20] * len(tie_players),
        'ExpectedRosterGain': [1.0] * len(tie_players),
    })
    tie_prelim = pd.DataFrame({
        'player': tie_players,
        'PrelimGain': [3.0] * len(tie_players),
        'PrelimSE': [1.0] * len(tie_players),
        'PrelimBlocks': [8] * len(tie_players),
        'PrelimN': [40] * len(tie_players),
        'PrelimAttempts': [40] * len(tie_players),
        'PrelimPairedRate': [100.0] * len(tie_players),
    })
    evidence_ties = simulation.target_screen_confirmation_shortlist(
        tie_base.sample(frac=1.0, random_state=19),
        tie_prelim.sample(frac=1.0, random_state=20),
        limit=10,
    )
    assert evidence_ties == sorted(tie_players)[:10]

    display_rows = pd.DataFrame({
        'player': [f'C{idx:02d}' for idx in range(35)],
        'EvidenceTier': ['Organic'] * 35,
        'EvidenceTierOrder': [2] * 35,
        'RankFamily': ['Organic'] * 35,
        'RankFamilyOrder': [1] * 35,
        'RankSupportOrder': [0] * 35,
        'TargetRankScore': list(reversed(range(35))),
        'SelectionCounts': [1.0] * 35,
        'MarketConfirmationAnchor': [False] * 34 + [True],
    })
    displayed = FootballSimulation.finalize_target_ranking(
        display_rows,
        limit=30,
    )
    assert len(displayed) == 30
    assert 'C34' in set(displayed.player)
    assert 'C29' not in set(displayed.player)
    assert displayed.iloc[-1].TargetOrder == 30


def verify_balanced_blocks():
    assert FootballSimulation.target_balanced_trial_blocks(320) == [40] * 8
    assert FootballSimulation.target_balanced_trial_blocks(300) == (
        [38] * 4 + [37] * 4
    )
    assert FootballSimulation.target_balanced_trial_blocks(4) == [1] * 4


def verify_dynamic_top_n():
    static = {
        'G_static': np.array([[-1.0, -1.0, 0.0, 0.0]]),
        'top_n_constraint_row': 0,
        'top_n_players': ('A', 'B'),
        'top_n_successor': 'C',
        'player_idx_map': {'A': 0, 'B': 1, 'C': 2, 'D': 3},
    }
    assert np.array_equal(
        FootballSimulation.target_pass_g_static(static, 'A')[0],
        [0.0, -1.0, -1.0, 0.0],
    )
    assert np.array_equal(
        FootballSimulation.target_pass_g_static(static, 'B')[0],
        [-1.0, 0.0, -1.0, 0.0],
    )
    assert FootballSimulation.target_pass_g_static(static, 'C') is (
        static['G_static']
    )
    no_constraint = {**static, 'top_n_constraint_row': None}
    assert FootballSimulation.target_pass_g_static(no_constraint, 'A') is (
        static['G_static']
    )


def main():
    verify_continuous_ranking()
    verify_pilot_allocation()
    verify_confirmation_allocation()
    verify_balanced_blocks()
    verify_dynamic_top_n()
    print(
        'Continuous ranking, disagreement, balanced-block, discovery-pilot, '
        'confirmation-allocation, and dynamic '
        'Top-N checks passed.'
    )


if __name__ == '__main__':
    main()
