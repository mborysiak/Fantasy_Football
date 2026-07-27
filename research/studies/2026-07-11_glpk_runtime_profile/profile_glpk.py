import argparse
import cProfile
import io
import json
import pstats
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from cvxopt import matrix, spmatrix
from cvxopt.glpk import lp


STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
APP_DIR = REPO_ROOT.parent / 'Fantasy_Football_App' / 'app'
DB_PATH = APP_DIR / 'Simulation.sqlite3'
RESULTS_DIR = STUDY_DIR / 'results'

sys.path.insert(0, str(APP_DIR))
import zSim_Helper as zsim  # noqa: E402


YEAR = 2026
LEAGUE = 'beta'
PRED_VERSION = 'final_ensemble'
SALARY_SOURCE = 'pred'
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
POS_MIN = {pos: LINEUP[pos] for pos in ('QB', 'RB', 'WR', 'TE')}


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


class RuntimeInstrumentation:
    def __init__(self, nominee):
        self.nominee = nominee
        self.scope = 'setup'
        self.branch = 'other'
        self.full_player_count = None
        self.elapsed = defaultdict(list)
        self.captured = {}
        self.scenario_modulus = None
        self.scenario_calls = defaultdict(int)
        self.scenario_rosters = defaultdict(list)
        self.solve_status_counts = defaultdict(int)

    def record(self, component, elapsed):
        self.elapsed[(self.scope, self.branch, component)].append(float(elapsed))

    def rows(self):
        output = []
        for (scope, branch, component), values in sorted(self.elapsed.items()):
            output.append({
                'scope': scope,
                'branch': branch,
                'component': component,
                'calls': len(values),
                'total_seconds': float(np.sum(values)),
                'mean_ms': float(1000 * np.mean(values)),
                'p50_ms': float(1000 * percentile(values, 50)),
                'p90_ms': float(1000 * percentile(values, 90)),
            })
        return output

    def roster_reuse_rows(self):
        rows = []
        for (scope, branch), roster_keys in sorted(self.scenario_rosters.items()):
            signatures = [signature for _, signature in roster_keys]
            unique_scenario_rosters = len(set(roster_keys))
            rows.append({
                'scope': scope,
                'branch': branch,
                'successful_scenarios': len(roster_keys),
                'unique_rosters': len(set(signatures)),
                'unique_scenario_rosters': unique_scenario_rosters,
                'reusable_score_calls': len(roster_keys) - unique_scenario_rosters,
                'reusable_score_rate': float(
                    1 - unique_scenario_rosters / max(len(roster_keys), 1)
                ),
            })
        return rows

    def solve_status_rows(self):
        return [
            {
                'scope': scope,
                'branch': branch,
                'status': status,
                'calls': calls,
            }
            for (scope, branch, status), calls in sorted(
                self.solve_status_counts.items()
            )
        ]


@contextmanager
def instrument_runtime(instrumentation):
    cls = zsim.FootballSimulation
    original_matrix = zsim.matrix
    original_solve_descriptor = cls.__dict__['solve_ilp']
    original_solve = cls.solve_ilp
    original_scenario = cls._solve_managed_scenario
    original_salary_descriptor = cls.__dict__['create_G_salaries']
    original_salary = cls.create_G_salaries
    original_normalize_descriptor = cls.__dict__['normalize_salary_market']
    original_normalize = cls.normalize_salary_market
    original_static = cls.build_managed_ilp_static_matrices
    original_template = cls.sample_template_weekly_scores
    original_marginal_descriptor = cls.__dict__['managed_marginal_values']
    original_marginal = cls.managed_marginal_values
    original_multi_descriptor = cls.__dict__['managed_lineup_multi_context_scores']
    original_multi = cls.managed_lineup_multi_context_scores
    original_contribution = cls.managed_roster_buy_pass_contributions

    def timed_matrix(*args, **kwargs):
        start = time.perf_counter()
        result = original_matrix(*args, **kwargs)
        instrumentation.record('cvxopt_matrix_conversion', time.perf_counter() - start)
        return result

    def timed_solve(c, g, h, a, b):
        start = time.perf_counter()
        result = original_solve(c, g, h, a, b)
        instrumentation.record('glpk_ilp', time.perf_counter() - start)
        instrumentation.solve_status_counts[
            (instrumentation.scope, instrumentation.branch, result[0])
        ] += 1
        capture_key = (instrumentation.scope, instrumentation.branch)
        if capture_key not in instrumentation.captured:
            instrumentation.captured[capture_key] = {
                'c': np.asarray(c, dtype=float).copy(),
                'G': np.asarray(g, dtype=float).copy(),
                'h': np.asarray(h, dtype=float).copy(),
                'A': np.asarray(a, dtype=float).copy(),
                'b': np.asarray(b, dtype=float).copy(),
                'status': result[0],
                'x': None if result[1] is None else np.asarray(result[1], dtype=float).copy(),
            }
        return result

    def timed_scenario(self, *args, **kwargs):
        predictions = args[0]
        fixed_players = list(args[5])
        old_branch = instrumentation.branch
        if instrumentation.scope == 'nomination':
            if instrumentation.nominee in fixed_players:
                instrumentation.branch = 'buy'
            elif (
                instrumentation.full_player_count is not None
                and len(predictions) == instrumentation.full_player_count - 1
            ):
                instrumentation.branch = 'pass'
            else:
                instrumentation.branch = 'open'
        else:
            instrumentation.branch = 'target'
        call_key = (instrumentation.scope, instrumentation.branch)
        scenario_ordinal = instrumentation.scenario_calls[call_key]
        if instrumentation.scenario_modulus:
            scenario_ordinal %= instrumentation.scenario_modulus
        instrumentation.scenario_calls[call_key] += 1
        start = time.perf_counter()
        try:
            result = original_scenario(self, *args, **kwargs)
            if result is not None and 'selected_players' in result:
                instrumentation.scenario_rosters[call_key].append((
                    scenario_ordinal,
                    tuple(sorted(result['selected_players'])),
                ))
            return result
        finally:
            instrumentation.record(
                'managed_scenario_total',
                time.perf_counter() - start,
            )
            instrumentation.branch = old_branch

    def timed_salary(*args, **kwargs):
        start = time.perf_counter()
        result = original_salary(*args, **kwargs)
        instrumentation.record('create_G_salaries', time.perf_counter() - start)
        return result

    def timed_normalize(*args, **kwargs):
        start = time.perf_counter()
        result = original_normalize(*args, **kwargs)
        instrumentation.record('normalize_salary_market', time.perf_counter() - start)
        return result

    def timed_static(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_static(self, *args, **kwargs)
        instrumentation.record('build_static_matrices', time.perf_counter() - start)
        return result

    def timed_template(self, *args, **kwargs):
        start = time.perf_counter()
        result = original_template(self, *args, **kwargs)
        instrumentation.record('sample_weekly_template', time.perf_counter() - start)
        return result

    def timed_marginal(cls_arg, *args, **kwargs):
        start = time.perf_counter()
        result = original_marginal(*args, **kwargs)
        instrumentation.record('managed_marginal_values', time.perf_counter() - start)
        return result

    def timed_multi(cls_arg, *args, **kwargs):
        start = time.perf_counter()
        result = original_multi(*args, **kwargs)
        instrumentation.record('multi_context_lineup_score', time.perf_counter() - start)
        return result

    def timed_contribution(self, *args, **kwargs):
        old_branch = instrumentation.branch
        if instrumentation.scope == 'target':
            instrumentation.branch = 'target_contribution'
        try:
            return original_contribution(self, *args, **kwargs)
        finally:
            instrumentation.branch = old_branch

    zsim.matrix = timed_matrix
    cls.solve_ilp = staticmethod(timed_solve)
    cls._solve_managed_scenario = timed_scenario
    cls.create_G_salaries = staticmethod(timed_salary)
    cls.normalize_salary_market = staticmethod(timed_normalize)
    cls.build_managed_ilp_static_matrices = timed_static
    cls.sample_template_weekly_scores = timed_template
    cls.managed_marginal_values = classmethod(timed_marginal)
    cls.managed_lineup_multi_context_scores = classmethod(timed_multi)
    cls.managed_roster_buy_pass_contributions = timed_contribution
    try:
        yield
    finally:
        zsim.matrix = original_matrix
        cls.solve_ilp = original_solve_descriptor
        cls._solve_managed_scenario = original_scenario
        cls.create_G_salaries = original_salary_descriptor
        cls.normalize_salary_market = original_normalize_descriptor
        cls.build_managed_ilp_static_matrices = original_static
        cls.sample_template_weekly_scores = original_template
        cls.managed_marginal_values = original_marginal_descriptor
        cls.managed_lineup_multi_context_scores = original_multi_descriptor
        cls.managed_roster_buy_pass_contributions = original_contribution


def create_sim():
    conn = sqlite3.connect(DB_PATH)
    sim = zsim.FootballSimulation(
        conn,
        YEAR,
        LINEUP,
        SALARY_CAP,
        PRED_VERSION,
        LEAGUE,
        sal_pred_actual=SALARY_SOURCE,
    )
    return conn, sim


def load_market_state(conn):
    keepers = pd.read_sql_query(
        """
        SELECT player, keeper_salary
        FROM League_Keepers
        WHERE year = ? AND league = ?
        """,
        conn,
        params=(YEAR, LEAGUE),
    )
    return {
        'to_add': {'players': [], 'salaries': []},
        'to_drop': keepers.player.tolist(),
        'remaining_market_budget': (
            NUM_TEAMS * SALARY_CAP - float(keepers.keeper_salary.sum())
        ),
        'remaining_market_slots': NUM_TEAMS * ROSTER_SIZE - len(keepers),
        'keepers': keepers,
    }


def profile_call(name, callback):
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = callback()
    profiler.disable()
    elapsed = time.perf_counter() - start
    profiler.dump_stats(RESULTS_DIR / f'{name}.prof')
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative')
    stats.print_stats(80)
    (RESULTS_DIR / f'{name}_cprofile.txt').write_text(stream.getvalue(), encoding='utf-8')
    return result, elapsed


def dense_matrix(array):
    return matrix(np.asarray(array, dtype=float), tc='d')


def sparse_matrix(array):
    array = np.asarray(array, dtype=float)
    rows, cols = np.nonzero(array)
    return spmatrix(
        array[rows, cols].tolist(),
        rows.tolist(),
        cols.tolist(),
        size=array.shape,
        tc='d',
    )


def remove_redundant_binary_lower_bounds(g, h):
    g = np.asarray(g, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)
    keep = np.ones(len(h), dtype=bool)
    removed = 0
    for row_idx in range(len(h)):
        nonzero = np.flatnonzero(np.abs(g[row_idx]) > 1e-12)
        if (
            h[row_idx] == 0
            and len(nonzero) == 1
            and g[row_idx, nonzero[0]] == -1
        ):
            keep[row_idx] = False
            removed += 1
    return g[keep], h[keep, None], removed


def solve_glpk(problem, g_array, h_array, sparse=False):
    c = dense_matrix(problem['c'])
    g = sparse_matrix(g_array) if sparse else dense_matrix(g_array)
    h = dense_matrix(h_array)
    a = dense_matrix(problem['A'])
    b = dense_matrix(problem['b'])
    return zsim.FootballSimulation.solve_ilp(c, g, h, a, b)


def objective(problem, x):
    if x is None:
        return np.nan
    return float(np.asarray(problem['c']).reshape(-1) @ np.asarray(x).reshape(-1))


def benchmark_glpk_variant(problem, g_array, h_array, repeats, sparse=False):
    c = dense_matrix(problem['c'])
    g = sparse_matrix(g_array) if sparse else dense_matrix(g_array)
    h = dense_matrix(h_array)
    a = dense_matrix(problem['A'])
    b = dense_matrix(problem['b'])
    zsim.FootballSimulation.solve_ilp(c, g, h, a, b)
    times = []
    status = None
    x = None
    for _ in range(repeats):
        start = time.perf_counter()
        status, x = zsim.FootballSimulation.solve_ilp(c, g, h, a, b)
        times.append(time.perf_counter() - start)
    return {
        'status': status,
        'objective': objective(problem, x),
        'mean_ms': float(1000 * np.mean(times)),
        'p50_ms': float(1000 * percentile(times, 50)),
        'p90_ms': float(1000 * percentile(times, 90)),
        'constraint_rows': int(np.asarray(g_array).shape[0]),
        'constraint_columns': int(np.asarray(g_array).shape[1]),
        'constraint_density': float(np.count_nonzero(g_array) / np.size(g_array)),
        '_selected': np.flatnonzero(
            np.asarray(x, dtype=float).reshape(-1) > 0.5
        ).tolist(),
    }


def benchmark_glpk_options(
    problem,
    g_array,
    h_array,
    repeats,
    option_overrides,
):
    original_options = dict(zsim.cvxopt.glpk.options)
    try:
        zsim.cvxopt.glpk.options.clear()
        zsim.cvxopt.glpk.options.update(original_options)
        zsim.cvxopt.glpk.options.update(option_overrides)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return benchmark_glpk_variant(
                problem,
                g_array,
                h_array,
                repeats,
                sparse=False,
            )
    finally:
        zsim.cvxopt.glpk.options.clear()
        zsim.cvxopt.glpk.options.update(original_options)


def benchmark_fresh_dense_conversion(problem, repeats):
    times = []
    status = None
    x = None
    for _ in range(repeats):
        start = time.perf_counter()
        status, x = solve_glpk(problem, problem['G'], problem['h'], sparse=False)
        times.append(time.perf_counter() - start)
    return {
        'status': status,
        'objective': objective(problem, x),
        'mean_ms': float(1000 * np.mean(times)),
        'p50_ms': float(1000 * percentile(times, 50)),
        'p90_ms': float(1000 * percentile(times, 90)),
        '_selected': np.flatnonzero(
            np.asarray(x, dtype=float).reshape(-1) > 0.5
        ).tolist(),
    }


def benchmark_lp_relaxation(problem, repeats):
    c = dense_matrix(problem['c'])
    g = dense_matrix(problem['G'])
    h = dense_matrix(problem['h'])
    a = dense_matrix(problem['A'])
    b = dense_matrix(problem['b'])
    lp(c, g, h, a, b)
    times = []
    status = None
    x = None
    for _ in range(repeats):
        start = time.perf_counter()
        status, x, _, _ = lp(c, g, h, a, b)
        times.append(time.perf_counter() - start)
    return {
        'status': status,
        'objective': objective(problem, x),
        'mean_ms': float(1000 * np.mean(times)),
        'p50_ms': float(1000 * percentile(times, 50)),
        'p90_ms': float(1000 * percentile(times, 90)),
        '_selected': None,
    }


def benchmark_highs(problem, g_array, h_array, repeats):
    from scipy.optimize import Bounds, LinearConstraint, milp

    c = np.asarray(problem['c'], dtype=float).reshape(-1)
    g = np.asarray(g_array, dtype=float)
    h = np.asarray(h_array, dtype=float).reshape(-1)
    a = np.asarray(problem['A'], dtype=float)
    b = np.asarray(problem['b'], dtype=float).reshape(-1)
    constraints = LinearConstraint(
        np.vstack([g, a]),
        np.concatenate([np.full(len(h), -np.inf), b]),
        np.concatenate([h, b]),
    )
    bounds = Bounds(np.zeros(len(c)), np.ones(len(c)))
    integrality = np.ones(len(c), dtype=int)

    def solve():
        return milp(
            c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={'disp': False},
        )

    solve()
    times = []
    result = None
    for _ in range(min(repeats, 30)):
        start = time.perf_counter()
        result = solve()
        times.append(time.perf_counter() - start)
    return {
        'status': int(result.status),
        'success': bool(result.success),
        'objective': float(result.fun),
        'mean_ms': float(1000 * np.mean(times)),
        'p50_ms': float(1000 * percentile(times, 50)),
        'p90_ms': float(1000 * percentile(times, 90)),
        '_selected': np.flatnonzero(result.x > 0.5).tolist(),
    }


def benchmark_captured_problems(captured, repeats, include_highs=False):
    rows = []
    for (scope, branch), problem in sorted(captured.items()):
        if problem['status'] != 'optimal':
            continue
        reduced_g, reduced_h, removed = remove_redundant_binary_lower_bounds(
            problem['G'],
            problem['h'],
        )
        variants = {
            'glpk_dense_reused': benchmark_glpk_variant(
                problem, problem['G'], problem['h'], repeats, sparse=False
            ),
            'glpk_sparse_reused': benchmark_glpk_variant(
                problem, problem['G'], problem['h'], repeats, sparse=True
            ),
            'glpk_dense_reduced': benchmark_glpk_variant(
                problem, reduced_g, reduced_h, repeats, sparse=False
            ),
            'glpk_sparse_reduced': benchmark_glpk_variant(
                problem, reduced_g, reduced_h, repeats, sparse=True
            ),
            'glpk_dense_fresh_conversion': benchmark_fresh_dense_conversion(
                problem, repeats
            ),
            'glpk_lp_relaxation': benchmark_lp_relaxation(problem, repeats),
        }
        if include_highs:
            variants['highs_reduced'] = benchmark_highs(
                problem, reduced_g, reduced_h, repeats
            )
        option_variants = {
            'glpk_presolve_on': {'presolve': True},
            'glpk_presolve_off': {'presolve': False},
            'glpk_branch_most_fractional': {'br_tech': 'GLP_BR_MFV'},
            'glpk_branch_drtom': {'br_tech': 'GLP_BR_DTH'},
            'glpk_branch_pseudocost': {'br_tech': 'GLP_BR_PCH'},
            'glpk_backtrack_best_local': {'bt_tech': 'GLP_BT_BLB'},
            'glpk_backtrack_best_projection': {'bt_tech': 'GLP_BT_BPH'},
            'glpk_mip_gap_0_005': {'mip_gap': 0.005},
        }
        for variant, options in option_variants.items():
            variants[variant] = benchmark_glpk_options(
                problem,
                problem['G'],
                problem['h'],
                repeats,
                options,
            )
        baseline_objective = variants['glpk_dense_reused']['objective']
        baseline_selected = variants['glpk_dense_reused']['_selected']
        for variant, metrics in variants.items():
            selected = metrics.pop('_selected')
            rows.append({
                'scope': scope,
                'branch': branch,
                'variant': variant,
                'removed_redundant_rows': removed,
                'objective_difference': float(
                    metrics['objective'] - baseline_objective
                ),
                'selected_difference_count': (
                    None
                    if selected is None
                    else len(set(selected) ^ set(baseline_selected))
                ),
                **metrics,
            })
    return rows


def benchmark_preparation(
    sim,
    to_drop,
    nominee,
    remaining_market_budget,
    remaining_market_slots,
    repeats=500,
):
    with sim.temp_seed(20260711):
        predictions = sim.drop_players(
            sim.get_predictions('pred_fp_per_game', num_options=1000),
            to_drop,
        )
        salary_samples = sim.drop_players(
            sim.get_salaries(num_options=1000),
            to_drop,
        )
    predictions = predictions.copy()
    predictions['salary'] = salary_samples.iloc[:, 2:7].mean(axis=1).to_numpy()
    fixed = {'players': [nominee], 'salaries': [float(predictions.loc[predictions.player == nominee, 'salary'].iloc[0])]}
    fixed_idx = np.flatnonzero(predictions.player.to_numpy() == nominee)
    salary_values = predictions.salary.to_numpy(dtype=float)
    fixed_mask = predictions.player.to_numpy() == nominee
    post_market_budget = remaining_market_budget - fixed['salaries'][0]
    post_market_slots = remaining_market_slots - 1
    lean_predictions = predictions[['player', 'pos', 'salary']].copy()

    def current_salary_row():
        return sim.create_G_salaries(predictions, fixed)

    def numpy_salary_row():
        values = salary_values.copy()
        values[fixed_idx] = fixed['salaries']
        return values.reshape(1, -1)

    def large_prediction_copy():
        copied = predictions.copy()
        return copied.loc[copied.player != nominee].reset_index(drop=True)

    def pandas_normalize_lean():
        frame = lean_predictions.copy()
        return sim.normalize_salary_market(
            frame,
            [nominee],
            remaining_market_budget=post_market_budget,
            remaining_market_slots=post_market_slots,
        ).salary.to_numpy()

    def pandas_normalize_full():
        frame = predictions.copy()
        return sim.normalize_salary_market(
            frame,
            [nominee],
            remaining_market_budget=post_market_budget,
            remaining_market_slots=post_market_slots,
        ).salary.to_numpy()

    def numpy_normalize():
        values = salary_values.copy()
        available_salaries = np.maximum(values[~fixed_mask], 1.0)
        top_idx = np.argpartition(
            available_salaries,
            len(available_salaries) - post_market_slots,
        )[-post_market_slots:]
        current_excess = float(np.sum(available_salaries[top_idx] - 1.0))
        target_excess = post_market_budget - post_market_slots
        scale = 0.0 if current_excess <= 0 else target_excess / current_excess
        values[~fixed_mask] = 1.0 + (available_salaries - 1.0) * scale
        return values

    benchmarks = {}
    for name, callback, count in (
        ('pandas_create_G_salaries', current_salary_row, repeats),
        ('numpy_salary_row', numpy_salary_row, repeats),
        ('copy_full_prediction_and_pass_subset', large_prediction_copy, min(repeats, 200)),
        ('pandas_normalize_lean', pandas_normalize_lean, repeats),
        ('pandas_normalize_full', pandas_normalize_full, min(repeats, 200)),
        ('numpy_normalize', numpy_normalize, repeats),
    ):
        callback()
        times = []
        for _ in range(count):
            start = time.perf_counter()
            callback()
            times.append(time.perf_counter() - start)
        benchmarks[name] = {
            'calls': count,
            'mean_ms': float(1000 * np.mean(times)),
            'p50_ms': float(1000 * percentile(times, 50)),
            'p90_ms': float(1000 * percentile(times, 90)),
        }
    benchmarks['salary_row_exact'] = bool(
        np.array_equal(current_salary_row(), numpy_salary_row())
    )
    benchmarks['salary_normalization_max_abs_difference'] = float(
        np.max(np.abs(pandas_normalize_lean() - numpy_normalize()))
    )
    benchmarks['prediction_shape'] = list(predictions.shape)
    return benchmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nomination-iters', type=int, default=100)
    parser.add_argument('--target-iters', type=int, default=25)
    parser.add_argument('--repeat-solves', type=int, default=100)
    parser.add_argument('--nominee', default='Saquon Barkley')
    parser.add_argument('--glpk-time-limit-ms', type=int, default=0)
    parser.add_argument('--skip-solver-microbench', action='store_true')
    parser.add_argument('--include-highs', action='store_true')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.glpk_time_limit_ms > 0:
        zsim.cvxopt.glpk.options['tm_lim'] = args.glpk_time_limit_ms
    instrumentation = RuntimeInstrumentation(args.nominee)
    instrumentation.scenario_modulus = args.nomination_iters

    conn, sim = create_sim()
    market = load_market_state(conn)
    baselines = sim.estimate_waiver_baselines(
        num_teams=NUM_TEAMS,
        roster_size=ROSTER_SIZE,
    )
    available_count = int(
        (~sim.player_data.player.isin(market['to_drop'])).sum()
    )
    instrumentation.full_player_count = available_count
    nominee_salary = int(round(float(
        sim.player_data.loc[sim.player_data.player == args.nominee, 'salary'].iloc[0]
    )))

    with instrument_runtime(instrumentation):
        instrumentation.scope = 'nomination'
        instrumentation.branch = 'setup'
        nomination, nomination_elapsed = profile_call(
            'nomination',
            lambda: sim.evaluate_nomination(
                market['to_add'],
                market['to_drop'],
                args.nominee,
                nominee_salary,
                num_iters=args.nomination_iters,
                require_top_n=12,
                next_year_frac=0,
                enforce_top_n=True,
                roster_size=ROSTER_SIZE,
                lineup_require=LINEUP,
                pos_min_counts=POS_MIN,
                pos_max_counts=zsim.MANAGED_POS_MAX,
                waiver_baselines=baselines,
                bench_upside_weight=0.25,
                remaining_market_budget=market['remaining_market_budget'],
                remaining_market_slots=market['remaining_market_slots'],
            ),
        )
        print(f'nomination profile complete: {nomination_elapsed:.3f}s', flush=True)

        conn.close()
        conn, target_sim = create_sim()
        instrumentation.scope = 'target'
        instrumentation.branch = 'target'
        target, target_elapsed = profile_call(
            'target',
            lambda: target_sim.run_sim(
                market['to_add'],
                market['to_drop'],
                args.target_iters,
                require_top_n=12,
                num_avg_pts=5,
                next_year_frac=0,
                enforce_top_n=True,
                scoring_mode='managed',
                roster_size=ROSTER_SIZE,
                lineup_require=LINEUP,
                pos_min_counts=POS_MIN,
                pos_max_counts=zsim.MANAGED_POS_MAX,
                waiver_baselines=baselines,
                bench_upside_weight=0.25,
                managed_holdout_contexts=5,
                remaining_market_budget=market['remaining_market_budget'],
                remaining_market_slots=market['remaining_market_slots'],
            ),
        )
        print(f'target profile complete: {target_elapsed:.3f}s', flush=True)

    instrumentation.scope = 'microbenchmark'
    instrumentation.branch = 'setup'
    preparation = benchmark_preparation(
        target_sim,
        market['to_drop'],
        args.nominee,
        market['remaining_market_budget'],
        market['remaining_market_slots'],
    )
    print('preparation benchmarks complete', flush=True)
    conn.close()

    solver_rows = []
    if not args.skip_solver_microbench:
        solver_rows = benchmark_captured_problems(
            instrumentation.captured,
            args.repeat_solves,
            include_highs=args.include_highs,
        )
        print('solver microbenchmarks complete', flush=True)
    for (scope, branch), problem in instrumentation.captured.items():
        np.savez_compressed(
            RESULTS_DIR / f'captured_{scope}_{branch}.npz',
            c=problem['c'],
            G=problem['G'],
            h=problem['h'],
            A=problem['A'],
            b=problem['b'],
            x=problem['x'],
        )
    timing_rows = instrumentation.rows()
    roster_reuse_rows = instrumentation.roster_reuse_rows()
    solve_status_rows = instrumentation.solve_status_rows()
    pd.DataFrame(timing_rows).to_csv(
        RESULTS_DIR / 'component_timings.csv',
        index=False,
    )
    pd.DataFrame(solver_rows).to_csv(
        RESULTS_DIR / 'solver_microbenchmarks.csv',
        index=False,
    )
    pd.DataFrame(roster_reuse_rows).to_csv(
        RESULTS_DIR / 'roster_reuse.csv',
        index=False,
    )
    pd.DataFrame(solve_status_rows).to_csv(
        RESULTS_DIR / 'solve_status_counts.csv',
        index=False,
    )

    output = {
        'config': {
            'nomination_iters': args.nomination_iters,
            'target_iters': args.target_iters,
            'repeat_solves': args.repeat_solves,
            'nominee': args.nominee,
            'nominee_salary': nominee_salary,
            'available_players': available_count,
            'keepers': len(market['keepers']),
            'glpk_time_limit_ms': args.glpk_time_limit_ms,
        },
        'nomination': {
            'elapsed_seconds': nomination_elapsed,
            'buy_edge': nomination['buy_edge'],
            'fair_bid': nomination['fair_bid'],
            'curve_points': len(nomination['price_curve']),
            'trials': nomination['trials'],
        },
        'target': {
            'elapsed_seconds': target_elapsed,
            'summary': target_sim.get_managed_summary(),
            'rows': len(target),
        },
        'preparation_microbenchmarks': preparation,
        'component_timings': timing_rows,
        'roster_reuse': roster_reuse_rows,
        'solve_status_counts': solve_status_rows,
        'solver_microbenchmarks': solver_rows,
    }
    (RESULTS_DIR / 'profile_summary.json').write_text(
        json.dumps(output, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({
        'nomination_seconds': round(nomination_elapsed, 3),
        'target_seconds': round(target_elapsed, 3),
        'timing_rows': len(timing_rows),
        'solver_rows': len(solver_rows),
        'results_dir': str(RESULTS_DIR),
    }, indent=2))


if __name__ == '__main__':
    main()
