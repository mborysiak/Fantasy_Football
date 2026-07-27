"""Measure sequential Bijan evidence stability across seeds and AJ states."""

from collections import Counter
from itertools import combinations
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parents[2]
APP_DIR = ROOT.parent / 'Fantasy_Football_App' / 'app'
RESULTS_DIR = STUDY_DIR / 'results'
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import zSequential_Target as sequential  # noqa: E402
from zSim_Helper import (  # noqa: E402
    FootballSimulation,
    MANAGED_POS_MAX,
)


YEAR = 2026
LEAGUE = 'beta'
PRED_VERSION = 'final_ensemble'
SALARY_SOURCE = 'pred'
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP_REQUIRE = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
POS_MIN = {pos: LINEUP_REQUIRE[pos] for pos in ('QB', 'RB', 'WR', 'TE')}
POS_MAX = MANAGED_POS_MAX.copy()
CONSTRUCTION_CONTEXTS = 32
CONFIRM_PATHS = 48
CONFIRM_SEASONS = 64
NUM_WEEKS = 16
LEARN_WEEKS = 6
MAX_LEARN_WEIGHT = 0.65
CANDIDATE = 'Bijan Robinson'
PRODUCTION_SEED = 3962081362

FIXED_SALARIES = {
    'Jahmyr Gibbs': 111.0,
    'Chase Brown': 34.0,
    'Bhayshul Tuten': 11.0,
}
AJ_SALE = {'Aj Brown': 53.0}
LEAGUE_KEEPER_SALARIES = {}
WAIVER_BASELINES = {}


def evidence_seeds(count=16):
    children = np.random.SeedSequence(20260720).spawn(max(0, count - 1))
    generated = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    ]
    return [PRODUCTION_SEED, *generated]


def draft_state(aj_available):
    unavailable = {
        player: salary
        for player, salary in LEAGUE_KEEPER_SALARIES.items()
        if player not in FIXED_SALARIES
    }
    if not aj_available:
        unavailable.update(AJ_SALE)
    nonkeeper_sales = {
        player: salary
        for player, salary in FIXED_SALARIES.items()
        if player not in LEAGUE_KEEPER_SALARIES
    }
    if not aj_available:
        nonkeeper_sales.update(AJ_SALE)
    remaining_budget = (
        NUM_TEAMS * SALARY_CAP
        - sum(LEAGUE_KEEPER_SALARIES.values())
        - sum(nonkeeper_sales.values())
    )
    remaining_slots = (
        NUM_TEAMS * ROSTER_SIZE
        - len(LEAGUE_KEEPER_SALARIES)
        - len(nonkeeper_sales)
    )
    return {
        'fixed': dict(FIXED_SALARIES),
        'unavailable': unavailable,
        'remaining_budget': float(remaining_budget),
        'remaining_slots': int(remaining_slots),
    }


def build_simulation():
    conn = sqlite3.connect(APP_DIR / 'Simulation.sqlite3')
    sim = FootballSimulation(
        conn,
        YEAR,
        LINEUP_REQUIRE,
        SALARY_CAP,
        PRED_VERSION,
        LEAGUE,
        sal_pred_actual=SALARY_SOURCE,
    )
    sim.load_weekly_template_profiles()
    return conn, sim


def validate_players(sim):
    required = {
        CANDIDATE,
        *FIXED_SALARIES,
        *LEAGUE_KEEPER_SALARIES,
        *AJ_SALE,
    }
    missing = sorted(required - set(sim.player_data.player))
    if missing:
        raise ValueError('Players missing from app database: ' + ', '.join(missing))


def _state_arrays(sim, full_predictions, state):
    full_players = full_predictions.player.to_numpy()
    unavailable = set(state['unavailable'])
    keep_mask = ~np.isin(full_players, list(unavailable))
    predictions = full_predictions.loc[keep_mask].reset_index(drop=True)
    state_full_indices = np.flatnonzero(keep_mask)

    full_source = sequential._aligned_player_frame(sim, full_predictions)
    full_market_prices = full_source.salary.to_numpy(dtype=np.float64)
    available_market_mask = ~np.isin(
        full_players,
        [*state['fixed'], *state['unavailable']],
    )
    normalized_full_prices = sim.normalize_salary_market_values(
        full_market_prices,
        available_market_mask,
        remaining_market_budget=state['remaining_budget'],
        remaining_market_slots=state['remaining_slots'],
    )
    base_prices = normalized_full_prices[keep_mask]
    market_prices = full_market_prices[keep_mask]
    predictions['salary'] = market_prices
    selection_premiums = sim.get_selection_premium_values(
        predictions.player.to_numpy(),
        fixed_players=list(state['fixed']),
        enabled=True,
    )
    return {
        'predictions': predictions,
        'keep_mask': keep_mask,
        'state_full_indices': state_full_indices,
        'base_prices': base_prices,
        'market_prices': market_prices,
        'selection_premiums': selection_premiums,
        'full_market_prices': full_market_prices,
        'available_market_mask': available_market_mask,
    }


def _nested_managed_values(sim, full_predictions, arrays, random_seed):
    predictions = arrays['predictions']
    keep_mask = arrays['keep_mask']
    preseason_ppg = predictions[
        sim.sample_value_columns(predictions)
    ].mean(axis=1)
    values = []
    with sim.temp_seed(random_seed):
        for _ in range(CONSTRUCTION_CONTEXTS):
            full_weekly, full_played = sim.sample_template_weekly_scores(
                full_predictions,
                num_weeks=NUM_WEEKS,
                return_played_mask=True,
            )
            weekly = full_weekly[keep_mask]
            played = full_played[keep_mask]
            decision = sim.build_managed_decision_scores(
                weekly,
                preseason_ppg=preseason_ppg,
                learn_weeks=LEARN_WEEKS,
                max_learn_weight=MAX_LEARN_WEIGHT,
                played_mask=played,
            )
            values.append(sim.managed_marginal_values(
                weekly,
                predictions.pos.to_numpy(),
                decision,
                predictions.player.to_numpy(),
                base_players=list(FIXED_SALARIES),
                waiver_baselines=WAIVER_BASELINES,
                lineup_require=LINEUP_REQUIRE,
                played_mask=played,
            ))
    return np.column_stack(values).mean(axis=1)


def _nested_tapes(sim, full_predictions, arrays, state, random_seed):
    full_players = full_predictions.player.to_numpy()
    state_players = arrays['predictions'].player.to_numpy()
    state_index = {player: idx for idx, player in enumerate(state_players)}
    with sim.temp_seed(random_seed):
        salaries = sim.get_salaries(num_options=CONFIRM_PATHS)
    salaries = (
        salaries.set_index('player').reindex(full_players).drop(columns=['pos'])
    )
    raw_prices = salaries.to_numpy(dtype=np.float64).T
    normalized_full = np.empty_like(raw_prices)
    for path_idx, row in enumerate(raw_prices):
        normalized_full[path_idx] = sim.normalize_salary_market_values(
            row,
            arrays['available_market_mask'],
            remaining_market_budget=state['remaining_budget'],
            remaining_market_slots=state['remaining_slots'],
        )
    normalized = np.maximum(
        1,
        np.rint(normalized_full[:, arrays['keep_mask']]),
    ).astype(np.int16)

    full_orders = sequential.noisy_salary_orders(
        arrays['full_market_prices'],
        CONFIRM_PATHS,
        random_seed=random_seed + 1,
        noise=sequential.DEFAULT_NOMINATION_NOISE,
    )
    orders = np.empty((CONFIRM_PATHS, len(state_players)), dtype=np.int32)
    unavailable = set(state['unavailable'])
    for path_idx, full_order in enumerate(full_orders):
        ordered_players = [
            full_players[idx]
            for idx in full_order
            if full_players[idx] not in unavailable
        ]
        orders[path_idx] = [state_index[player] for player in ordered_players]
    return {'orders': orders, 'prices': normalized}


def _nested_validation_bank(sim, full_predictions, arrays, random_seed):
    with sim.temp_seed(random_seed):
        full_bank = sim.sample_managed_holdout_contexts(
            full_predictions,
            num_contexts=CONFIRM_SEASONS,
            num_weeks=NUM_WEEKS,
            learn_weeks=LEARN_WEEKS,
            max_learn_weight=MAX_LEARN_WEIGHT,
            return_played_masks=True,
        )
    keep_mask = arrays['keep_mask']
    return tuple(context[:, keep_mask, :] for context in full_bank)


def _current_inputs(sim, root_seed, state):
    seed_values = [
        int(seed.generate_state(1, dtype=np.uint32)[0])
        for seed in np.random.SeedSequence(int(root_seed)).spawn(8)
    ]
    with sim.temp_seed(seed_values[0]):
        predictions = sim.drop_players(
            sim.get_predictions('pred_fp_per_game', num_options=512),
            list(state['unavailable']),
        )
    aligned = sequential._aligned_player_frame(sim, predictions)
    market_prices = aligned.salary.to_numpy(dtype=np.float64)
    available_mask = ~predictions.player.isin(state['fixed']).to_numpy()
    base_prices = sim.normalize_salary_market_values(
        market_prices,
        available_mask,
        remaining_market_budget=state['remaining_budget'],
        remaining_market_slots=state['remaining_slots'],
    )
    predictions['salary'] = market_prices
    selection_premiums = sim.get_selection_premium_values(
        predictions.player.to_numpy(),
        fixed_players=list(state['fixed']),
        enabled=True,
    )
    with sim.temp_seed(seed_values[1]):
        managed_matrix = sim.sample_managed_value_matrix(
            predictions,
            list(state['fixed']),
            num_options=CONSTRUCTION_CONTEXTS,
            num_weeks=NUM_WEEKS,
            waiver_baselines=WAIVER_BASELINES,
            lineup_require=LINEUP_REQUIRE,
            learn_weeks=LEARN_WEEKS,
            max_learn_weight=MAX_LEARN_WEIGHT,
        )
    tapes = sequential.generate_hidden_auction_tapes(
        sim,
        predictions,
        state['fixed'],
        CONFIRM_PATHS,
        state['remaining_budget'],
        state['remaining_slots'],
        seed_values[5],
    )
    validation_bank = sequential._sample_validation_bank(
        sim,
        predictions,
        CONFIRM_SEASONS,
        NUM_WEEKS,
        LEARN_WEEKS,
        MAX_LEARN_WEIGHT,
        seed_values[6],
    )
    return {
        'seed_values': seed_values,
        'predictions': predictions,
        'managed_values': managed_matrix.mean(axis=1),
        'base_prices': base_prices,
        'market_prices': market_prices,
        'selection_premiums': selection_premiums,
        'tapes': tapes,
        'validation_bank': validation_bank,
    }


def _nested_inputs(sim, root_seed, state):
    return _nested_inputs_with_plan(sim, root_seed, state)


def _seed_values(root_seed):
    return [
        int(seed.generate_state(1, dtype=np.uint32)[0])
        for seed in np.random.SeedSequence(int(root_seed)).spawn(8)
    ]


def _nested_inputs_with_plan(
    sim,
    root_seed,
    state,
    construction_root=None,
    tape_root=None,
    validation_root=None,
):
    seed_values = _seed_values(root_seed)
    construction_seeds = _seed_values(
        root_seed if construction_root is None else construction_root
    )
    tape_seeds = _seed_values(root_seed if tape_root is None else tape_root)
    validation_seeds = _seed_values(
        root_seed if validation_root is None else validation_root
    )
    with sim.temp_seed(construction_seeds[0]):
        full_predictions = sim.get_predictions(
            'pred_fp_per_game',
            num_options=512,
        )
    arrays = _state_arrays(sim, full_predictions, state)
    return {
        'seed_values': seed_values,
        'predictions': arrays['predictions'],
        'managed_values': _nested_managed_values(
            sim,
            full_predictions,
            arrays,
            construction_seeds[1],
        ),
        'base_prices': arrays['base_prices'],
        'market_prices': arrays['market_prices'],
        'selection_premiums': arrays['selection_premiums'],
        'tapes': _nested_tapes(
            sim,
            full_predictions,
            arrays,
            state,
            tape_seeds[5],
        ),
        'validation_bank': _nested_validation_bank(
            sim,
            full_predictions,
            arrays,
            validation_seeds[6],
        ),
    }


def evaluate_bijan(
    sim,
    root_seed,
    aj_available,
    mode,
    construction_root=None,
    tape_root=None,
    validation_root=None,
):
    state = draft_state(aj_available)
    started = time.perf_counter()
    inputs = (
        _current_inputs(sim, root_seed, state)
        if mode == 'current'
        else _nested_inputs_with_plan(
            sim,
            root_seed,
            state,
            construction_root=construction_root,
            tape_root=tape_root,
            validation_root=validation_root,
        )
    )
    predictions = inputs['predictions']
    player_idx = {
        player: idx for idx, player in enumerate(predictions.player)
    }
    candidate_idx = player_idx[CANDIDATE]
    candidate_price = sequential._round_price(
        inputs['market_prices'][candidate_idx]
    )
    captured = {}
    original_summary = sequential.summarize_sequential_differences

    def capture_summary(differences, *args, **kwargs):
        captured['differences'] = np.asarray(differences, dtype=np.float64)
        return original_summary(differences, *args, **kwargs)

    sequential.summarize_sequential_differences = capture_summary
    try:
        result = sequential.evaluate_sequential_candidate_price(
            sim,
            predictions,
            inputs['managed_values'],
            inputs['base_prices'],
            inputs['selection_premiums'],
            state['fixed'],
            CANDIDATE,
            candidate_price,
            inputs['tapes'],
            *inputs['validation_bank'],
            state['remaining_budget'],
            state['remaining_slots'],
            ROSTER_SIZE,
            LINEUP_REQUIRE,
            POS_MIN,
            POS_MAX,
            12,
            True,
            WAIVER_BASELINES,
            {},
            random_seed=(
                inputs['seed_values'][7]
                + 1009 * candidate_idx
                + candidate_price
            ),
        )
    finally:
        sequential.summarize_sequential_differences = original_summary

    differences = captured['differences']
    path_means = differences.mean(axis=1)
    season_means = differences.mean(axis=0)
    return {
        'mode': mode,
        'root_seed': int(root_seed),
        'aj_available': bool(aj_available),
        'gain': result['SequentialGain'],
        'se': result['SequentialSE'],
        'lcb80': result['SequentialLCB80'],
        'buy_ev': result['BuyEV'],
        'pass_ev': result['PassEV'],
        'paired_n': result['PairedN'],
        'buy_completion': result['BuyCompletion'],
        'pass_completion': result['PassCompletion'],
        'path_mean_sd': float(np.std(path_means, ddof=1)),
        'season_mean_sd': float(np.std(season_means, ddof=1)),
        'runtime_seconds': float(time.perf_counter() - started),
    }


def run_variance_decomposition(sim, seeds):
    scenarios = {
        'all_varied': lambda seed: (seed, seed, seed),
        'construction_only': lambda seed: (
            seed,
            PRODUCTION_SEED,
            PRODUCTION_SEED,
        ),
        'evidence_only': lambda seed: (
            PRODUCTION_SEED,
            seed,
            seed,
        ),
        'auction_paths_only': lambda seed: (
            PRODUCTION_SEED,
            seed,
            PRODUCTION_SEED,
        ),
        'weekly_seasons_only': lambda seed: (
            PRODUCTION_SEED,
            PRODUCTION_SEED,
            seed,
        ),
    }
    rows = []
    total = len(seeds) * len(scenarios)
    for scenario, plan_for_seed in scenarios.items():
        for root_seed in seeds:
            construction_root, tape_root, validation_root = plan_for_seed(
                root_seed
            )
            row = evaluate_bijan(
                sim,
                root_seed,
                aj_available=True,
                mode='nested',
                construction_root=construction_root,
                tape_root=tape_root,
                validation_root=validation_root,
            )
            row['scenario'] = scenario
            row['construction_root'] = construction_root
            row['tape_root'] = tape_root
            row['validation_root'] = validation_root
            rows.append(row)
            print(
                f"decomp {len(rows):02d}/{total} {scenario:19s} "
                f"seed={root_seed} gain={row['gain']:+.2f}"
            )
    return pd.DataFrame(rows)


def build_panel_stability(results):
    seed_rows = (
        results[
            (results['mode'] == 'nested')
            & results['aj_available']
        ]
        .sort_values('root_seed')
        .reset_index(drop=True)
    )
    rows = []
    for panel_size in (1, 2, 4, 8):
        for panel_id, indices in enumerate(
            combinations(range(len(seed_rows)), panel_size)
        ):
            panel = seed_rows.iloc[list(indices)]
            panel_gain = float(panel.gain.mean())
            panel_se = float(np.sqrt(np.square(panel.se).sum()) / panel_size)
            rows.append({
                'panel_size': panel_size,
                'panel_id': panel_id,
                'seeds': ','.join(map(str, panel.root_seed)),
                'gain': panel_gain,
                'se': panel_se,
                'lcb80': panel_gain - sequential.SEQUENTIAL_ACTION_LCB_Z * panel_se,
            })
    return pd.DataFrame(rows)


def write_summary(results, deltas, panels, decomposition):
    lines = [
        '# Sequential Seed Stability Results',
        '',
        '## Seed-level forced Bijan evidence',
        '',
    ]
    for (mode, aj_available), group in results.groupby(
        ['mode', 'aj_available'],
        sort=True,
    ):
        label = 'AJ available' if aj_available else 'AJ unavailable'
        lines.append(
            f'- **{mode}, {label}:** gain mean `{group.gain.mean():+.2f}`, '
            f'seed SD `{group.gain.std(ddof=1):.2f}`, range '
            f'`[{group.gain.min():+.2f}, {group.gain.max():+.2f}]`; '
            f'LCB80 positive in `{100 * (group.lcb80 > 0).mean():.1f}%` of seeds.'
        )
    lines.extend([
        '',
        '## AJ-off minus AJ-on within the same root seed',
        '',
    ])
    for mode, group in deltas.groupby('mode', sort=True):
        lines.append(
            f'- **{mode}:** correlation `{group.gain_on.corr(group.gain_off):.2f}`, '
            f'mean edge change `{group.delta_gain.mean():+.2f}`, SD '
            f'`{group.delta_gain.std(ddof=1):.2f}`, range '
            f'`[{group.delta_gain.min():+.2f}, {group.delta_gain.max():+.2f}]`.'
        )
    lines.extend([
        '',
        '## Independent evidence-bank panels (nested AJ-available state)',
        '',
    ])
    for panel_size, group in panels.groupby('panel_size', sort=True):
        lines.append(
            f'- **{panel_size} bank(s):** panel-gain SD '
            f'`{group.gain.std(ddof=1) if len(group) > 1 else 0.0:.2f}`, '
            f'range `[{group.gain.min():+.2f}, {group.gain.max():+.2f}]`; '
            f'LCB80 positive in `{100 * (group.lcb80 > 0).mean():.1f}%` of panels.'
        )
    lines.extend([
        '',
        '## Variance decomposition (nested AJ-available state)',
        '',
    ])
    for scenario, group in decomposition.groupby('scenario', sort=True):
        lines.append(
            f'- **{scenario}:** gain mean `{group.gain.mean():+.2f}`, '
            f'seed SD `{group.gain.std(ddof=1):.2f}`, range '
            f'`[{group.gain.min():+.2f}, {group.gain.max():+.2f}]`.'
        )
    lines.extend([
        '',
        'The production decision should use a fixed multi-bank, player-keyed '
        'evidence panel and add banks adaptively when independent bank estimates '
        'disagree or the action boundary remains unresolved. A single root seed '
        'should not be treated as a robustness guarantee.',
        '',
    ])
    (RESULTS_DIR / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    global LEAGUE_KEEPER_SALARIES
    global WAIVER_BASELINES
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn, sim = build_simulation()
    try:
        keepers = pd.read_sql_query(
            """
            SELECT player, keeper_salary
            FROM League_Keepers
            WHERE year = :year AND league = :league
            """,
            conn,
            params={'year': YEAR, 'league': LEAGUE},
        )
        LEAGUE_KEEPER_SALARIES = dict(zip(
            keepers.player,
            keepers.keeper_salary.astype(float),
        ))
        WAIVER_BASELINES = sim.estimate_waiver_baselines(
            num_teams=NUM_TEAMS,
            roster_size=ROSTER_SIZE,
        )
        print(
            f'Loaded {len(LEAGUE_KEEPER_SALARIES)} league keepers for '
            f'${sum(LEAGUE_KEEPER_SALARIES.values()):.0f}; '
            f'waiver baselines={WAIVER_BASELINES}.'
        )
        validate_players(sim)
        rows = []
        seeds = evidence_seeds()
        total = len(seeds) * 4
        for seed_index, root_seed in enumerate(seeds):
            for mode in ('current', 'nested'):
                for aj_available in (True, False):
                    row = evaluate_bijan(
                        sim,
                        root_seed,
                        aj_available,
                        mode,
                    )
                    rows.append(row)
                    print(
                        f"{len(rows):02d}/{total} {mode:7s} "
                        f"AJ={'on ' if aj_available else 'off'} "
                        f"seed={root_seed} gain={row['gain']:+.2f} "
                        f"LCB={row['lcb80']:+.2f}"
                    )
    finally:
        conn.close()

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / 'bijan_seed_results.csv', index=False)
    paired = results.pivot(
        index=['mode', 'root_seed'],
        columns='aj_available',
        values=['gain', 'lcb80', 'buy_ev', 'pass_ev'],
    )
    paired.columns = [
        f'{metric}_{"on" if available else "off"}'
        for metric, available in paired.columns
    ]
    paired = paired.reset_index()
    paired['delta_gain'] = paired.gain_off - paired.gain_on
    paired['delta_buy_ev'] = paired.buy_ev_off - paired.buy_ev_on
    paired['delta_pass_ev'] = paired.pass_ev_off - paired.pass_ev_on
    paired.to_csv(RESULTS_DIR / 'aj_state_deltas.csv', index=False)

    panels = build_panel_stability(results)
    panels.to_csv(RESULTS_DIR / 'panel_stability.csv', index=False)
    decomposition = run_variance_decomposition(sim, evidence_seeds())
    decomposition.to_csv(
        RESULTS_DIR / 'variance_decomposition.csv',
        index=False,
    )
    write_summary(results, paired, panels, decomposition)
    print(RESULTS_DIR / 'summary.md')


if __name__ == '__main__':
    main()
