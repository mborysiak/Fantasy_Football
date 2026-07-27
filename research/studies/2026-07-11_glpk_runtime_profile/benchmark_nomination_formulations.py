import argparse
import json
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
APP_DIR = REPO_ROOT.parent / 'Fantasy_Football_App' / 'app'
DB_PATH = APP_DIR / 'Simulation.sqlite3'
RESULTS_DIR = STUDY_DIR / 'results'
sys.path.insert(0, str(APP_DIR))
import zSim_Helper as zsim  # noqa: E402


LINEUP = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
POS_MIN = {pos: LINEUP[pos] for pos in ('QB', 'RB', 'WR', 'TE')}


def remove_redundant_rows(g, h):
    keep = np.ones(len(h), dtype=bool)
    for row_idx in range(len(h)):
        nonzero = np.flatnonzero(np.abs(g[row_idx]) > 1e-12)
        if (
            float(h[row_idx, 0]) == 0
            and len(nonzero) == 1
            and float(g[row_idx, nonzero[0]]) == -1
        ):
            keep[row_idx] = False
    return g[keep], h[keep]


@contextmanager
def reduced_static_constraints(enabled):
    cls = zsim.FootballSimulation
    original = cls.build_managed_ilp_static_matrices

    def reduced(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        output['G_static'], output['h_static'] = remove_redundant_rows(
            output['G_static'],
            output['h_static'],
        )
        return output

    if enabled:
        cls.build_managed_ilp_static_matrices = reduced
    try:
        yield
    finally:
        cls.build_managed_ilp_static_matrices = original


def run_variant(name, num_iters, backtrack_projection, reduce_rows):
    original_options = dict(zsim.cvxopt.glpk.options)
    zsim.cvxopt.glpk.options.clear()
    zsim.cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
    zsim.cvxopt.glpk.options['tm_lim'] = 100
    if backtrack_projection:
        zsim.cvxopt.glpk.options['bt_tech'] = 'GLP_BT_BPH'

    conn = sqlite3.connect(DB_PATH)
    try:
        sim = zsim.FootballSimulation(
            conn,
            2026,
            LINEUP,
            298,
            'final_ensemble',
            'beta',
            sal_pred_actual='pred',
        )
        keepers = pd.read_sql_query(
            """
            SELECT player, keeper_salary
            FROM League_Keepers
            WHERE year = 2026 AND league = 'beta'
            """,
            conn,
        )
        start = time.perf_counter()
        with reduced_static_constraints(reduce_rows):
            result = sim.evaluate_nomination(
                {'players': [], 'salaries': []},
                keepers.player.tolist(),
                'Saquon Barkley',
                76,
                num_iters=num_iters,
                require_top_n=12,
                next_year_frac=0,
                enforce_top_n=True,
                roster_size=13,
                lineup_require=LINEUP,
                pos_min_counts=POS_MIN,
                pos_max_counts=zsim.MANAGED_POS_MAX,
                waiver_baselines=sim.estimate_waiver_baselines(12, 13),
                bench_upside_weight=0.25,
                remaining_market_budget=(
                    12 * 298 - float(keepers.keeper_salary.sum())
                ),
                remaining_market_slots=12 * 13 - len(keepers),
            )
        elapsed = time.perf_counter() - start
    finally:
        conn.close()
        zsim.cvxopt.glpk.options.clear()
        zsim.cvxopt.glpk.options.update(original_options)

    curve = result['price_curve'].sort_values('price').reset_index(drop=True)
    return {
        'name': name,
        'elapsed_seconds': elapsed,
        'buy_edge': result['buy_edge'],
        'fair_bid': result['fair_bid'],
        'fit_rate': result['fit_rate'],
        'buy_win_rate': result['buy_win_rate'],
        'curve': curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-iters', type=int, default=100)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        ('baseline', False, False),
        ('best_projection', True, False),
        ('reduced_rows', False, True),
        ('best_projection_reduced_rows', True, True),
    ]
    results = [
        run_variant(name, args.num_iters, backtrack, reduced)
        for name, backtrack, reduced in variants
    ]
    baseline = results[0]
    rows = []
    for result in results:
        baseline_curve = baseline['curve']
        curve = result['curve']
        same_prices = baseline_curve.price.tolist() == curve.price.tolist()
        numeric_columns = [
            'buy_ev', 'pass_ev', 'buy_edge', 'buy_win_rate',
            'expected_starts', 'fit_rate',
        ]
        max_curve_difference = np.nan
        if same_prices:
            max_curve_difference = float(np.nanmax(np.abs(
                baseline_curve[numeric_columns].to_numpy(dtype=float)
                - curve[numeric_columns].to_numpy(dtype=float)
            )))
        rows.append({
            'variant': result['name'],
            'elapsed_seconds': result['elapsed_seconds'],
            'speedup_vs_baseline': (
                baseline['elapsed_seconds'] / result['elapsed_seconds']
            ),
            'buy_edge': result['buy_edge'],
            'fair_bid': result['fair_bid'],
            'fit_rate': result['fit_rate'],
            'buy_win_rate': result['buy_win_rate'],
            'same_curve_prices': same_prices,
            'max_curve_difference': max_curve_difference,
        })
        print(json.dumps(rows[-1], indent=2), flush=True)

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / 'nomination_formulation_benchmark.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
