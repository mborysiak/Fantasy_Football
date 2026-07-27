"""Validate the production v4 multi-block sequential evidence panel."""

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
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import run_seed_stability as original  # noqa: E402
import zSequential_Target as sequential  # noqa: E402


BLOCKS = 4
SEEDS = original.evidence_seeds(16)


def production_v4_inputs(sim, root_seed, state):
    seed_values = original._seed_values(root_seed)
    with sim.temp_seed(seed_values[0]):
        canonical_predictions = sim.get_predictions(
            'pred_fp_per_game',
            num_options=512,
        )
    arrays = original._state_arrays(sim, canonical_predictions, state)
    predictions = arrays['predictions']
    managed_blocks = sequential._sample_construction_value_blocks(
        sim,
        canonical_predictions,
        predictions,
        list(state['fixed']),
        block_count=BLOCKS,
        contexts_per_block=original.CONSTRUCTION_CONTEXTS,
        num_weeks=original.NUM_WEEKS,
        waiver_baselines=original.WAIVER_BASELINES,
        lineup_require=original.LINEUP_REQUIRE,
        learn_weeks=original.LEARN_WEEKS,
        max_learn_weight=original.MAX_LEARN_WEIGHT,
        random_seed=seed_values[1],
    )
    path_counts = sequential._split_evidence_counts(
        original.CONFIRM_PATHS,
        BLOCKS,
    )
    tape_seeds = np.random.SeedSequence(seed_values[5]).spawn(BLOCKS)
    bank_seeds = np.random.SeedSequence(seed_values[6]).spawn(BLOCKS)
    evidence_blocks = []
    for block_idx, path_count in enumerate(path_counts):
        tape_seed = int(
            tape_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
        )
        bank_seed = int(
            bank_seeds[block_idx].generate_state(1, dtype=np.uint32)[0]
        )
        evidence_blocks.append({
            'managed_values': managed_blocks[block_idx],
            'tapes': sequential.generate_hidden_auction_tapes(
                sim,
                predictions,
                state['fixed'],
                path_count,
                state['remaining_budget'],
                state['remaining_slots'],
                tape_seed,
                canonical_predictions=canonical_predictions,
            ),
            'validation_bank': sequential._sample_validation_bank(
                sim,
                predictions,
                original.CONFIRM_SEASONS,
                original.NUM_WEEKS,
                original.LEARN_WEEKS,
                original.MAX_LEARN_WEIGHT,
                bank_seed,
                canonical_predictions=canonical_predictions,
            ),
        })
    return arrays, evidence_blocks, seed_values


def evaluate(sim, root_seed, aj_available):
    state = original.draft_state(aj_available)
    started = time.perf_counter()
    arrays, evidence_blocks, seed_values = production_v4_inputs(
        sim,
        root_seed,
        state,
    )
    predictions = arrays['predictions']
    candidate_idx = predictions.index[
        predictions.player == original.CANDIDATE
    ][0]
    price = sequential._round_price(arrays['market_prices'][candidate_idx])
    result = sequential.evaluate_sequential_candidate_price_blocks(
        sim,
        predictions,
        arrays['base_prices'],
        arrays['selection_premiums'],
        state['fixed'],
        original.CANDIDATE,
        price,
        evidence_blocks,
        state['remaining_budget'],
        state['remaining_slots'],
        original.ROSTER_SIZE,
        original.LINEUP_REQUIRE,
        original.POS_MIN,
        original.POS_MAX,
        12,
        True,
        original.WAIVER_BASELINES,
        [{} for _ in range(BLOCKS)],
        random_seed=sequential.stable_sequential_component_seed(
            seed_values[7],
            'confirm',
            original.CANDIDATE,
            price,
        ),
    )
    return {
        'root_seed': int(root_seed),
        'aj_available': bool(aj_available),
        'gain': result['SequentialGain'],
        'se': result['SequentialSE'],
        'lcb80': result['SequentialLCB80'],
        'block_sd': result['BlockGainSD'],
        'block_min': result['BlockGainMin'],
        'block_max': result['BlockGainMax'],
        'block_positive_rate': result['BlockPositiveRate'],
        'buy_ev': result['BuyEV'],
        'pass_ev': result['PassEV'],
        'runtime_seconds': float(time.perf_counter() - started),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn, sim = original.build_simulation()
    try:
        keepers = pd.read_sql_query(
            """
            SELECT player, keeper_salary
            FROM League_Keepers
            WHERE year = :year AND league = :league
            """,
            conn,
            params={'year': original.YEAR, 'league': original.LEAGUE},
        )
        original.LEAGUE_KEEPER_SALARIES = dict(zip(
            keepers.player,
            keepers.keeper_salary.astype(float),
        ))
        original.WAIVER_BASELINES = sim.estimate_waiver_baselines(
            num_teams=original.NUM_TEAMS,
            roster_size=original.ROSTER_SIZE,
        )
        original.validate_players(sim)
        rows = []
        for root_seed in SEEDS:
            for aj_available in (True, False):
                row = evaluate(sim, root_seed, aj_available)
                rows.append(row)
                print(
                    f"seed={root_seed} AJ={aj_available} "
                    f"gain={row['gain']:+.2f} LCB={row['lcb80']:+.2f}"
                )
    finally:
        conn.close()

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / 'v4_bijan_seed_results.csv', index=False)
    paired = results.pivot(index='root_seed', columns='aj_available')
    deltas = pd.DataFrame({
        'gain_delta_off_minus_on': paired['gain'][False] - paired['gain'][True],
        'lcb_delta_off_minus_on': paired['lcb80'][False] - paired['lcb80'][True],
    }).reset_index()
    deltas.to_csv(RESULTS_DIR / 'v4_aj_state_deltas.csv', index=False)

    gain_summary = results.groupby('aj_available').gain.agg(
        ['mean', 'std', 'min', 'max']
    )
    correlation = float(
        paired['gain'][False].corr(paired['gain'][True])
    )
    lines = [
        '# Production v4 Stability Check',
        '',
        'Four independent blocks; each uses 32 balanced mean-PPG construction '
        'templates, 12 realized auction paths, and 64 complete validation seasons.',
        '',
    ]
    for aj_available, label in ((True, 'AJ available'), (False, 'AJ unavailable')):
        row = gain_summary.loc[aj_available]
        lines.append(
            f"- **{label}:** mean `{row['mean']:+.2f}`, seed SD "
            f"`{row['std']:.2f}`, range `[{row['min']:+.2f}, {row['max']:+.2f}]`."
        )
    lines.extend([
        f"- **AJ state correlation:** `{correlation:.3f}`.",
        f"- **AJ-off minus AJ-on:** mean "
        f"`{deltas.gain_delta_off_minus_on.mean():+.2f}`, SD "
        f"`{deltas.gain_delta_off_minus_on.std(ddof=1):.2f}`, range "
        f"`[{deltas.gain_delta_off_minus_on.min():+.2f}, "
        f"{deltas.gain_delta_off_minus_on.max():+.2f}]`.",
        '',
    ])
    (RESULTS_DIR / 'v4_summary.md').write_text(
        '\n'.join(lines),
        encoding='utf-8',
    )
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
