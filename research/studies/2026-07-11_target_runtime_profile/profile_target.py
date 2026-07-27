import argparse
import cProfile
import io
import json
import math
import pstats
import sqlite3
import sys
import time
from collections import Counter, defaultdict
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


YEAR = 2026
LEAGUE = 'beta'
PRED_VERSION = 'final_ensemble'
SALARY_SOURCE = 'pred'
SALARY_CAP = 298
NUM_TEAMS = 12
ROSTER_SIZE = 13
LINEUP = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 2}
POS_MIN = {pos: LINEUP[pos] for pos in ('QB', 'RB', 'WR', 'TE')}
SEED = 20260711


def percentile(values, quantile):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


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
        'keeper_count': len(keepers),
    }


def target_kwargs(sim, market):
    return {
        'require_top_n': 12,
        'num_avg_pts': 5,
        'next_year_frac': 0,
        'enforce_top_n': True,
        'scoring_mode': 'managed',
        'roster_size': ROSTER_SIZE,
        'lineup_require': LINEUP,
        'pos_min_counts': POS_MIN,
        'pos_max_counts': zsim.MANAGED_POS_MAX,
        'waiver_baselines': sim.estimate_waiver_baselines(
            NUM_TEAMS,
            ROSTER_SIZE,
        ),
        'bench_upside_weight': 0.25,
        'managed_value_options': 50,
        'managed_context_draws': 5,
        'managed_holdout_contexts': 5,
        'managed_context_refresh_interval': 50,
        'remaining_market_budget': market['remaining_market_budget'],
        'remaining_market_slots': market['remaining_market_slots'],
    }


def summarize_result(result, sim, elapsed, iterations):
    summary = sim.get_managed_summary() or {}
    return {
        'iterations': int(iterations),
        'elapsed_seconds': float(elapsed),
        'trials_per_second': float(iterations / elapsed),
        'result_rows': int(len(result)),
        'top_player': None if len(result) == 0 else str(result.player.iloc[0]),
        'top_expected_roster_gain': (
            None
            if len(result) == 0
            else float(result.ExpectedRosterGain.iloc[0])
        ),
        'managed_trials': int(summary.get('trials', 0)),
        'season_ev': float(summary.get('season_ev', np.nan)),
    }


def run_serial(iterations, seed=SEED, instrumentation=None):
    conn, sim = create_sim()
    try:
        market = load_market_state(conn)
        kwargs = target_kwargs(sim, market)
        start = time.perf_counter()
        with sim.temp_seed(seed):
            if instrumentation is None:
                result = sim.run_sim(
                    market['to_add'],
                    market['to_drop'],
                    iterations,
                    **kwargs,
                )
            else:
                with instrument_target(instrumentation):
                    result = sim.run_sim(
                        market['to_add'],
                        market['to_drop'],
                        iterations,
                        **kwargs,
                    )
        elapsed = time.perf_counter() - start
        return result, summarize_result(result, sim, elapsed, iterations)
    finally:
        conn.close()


class TargetInstrumentation:
    def __init__(self):
        self.phase = 'outer'
        self.elapsed = defaultdict(list)
        self.status_counts = Counter()
        self.current_contribution = None
        self.contribution_calls = 0
        self.contribution_solve_ordinal = 0
        self.candidate_evaluations = 0
        self.context_cache_checks = 0
        self.context_cache_hits = 0
        self.context_cache_misses = 0
        self.score_roster_keys = []
        self.selected_candidates = []
        self.unique_contexts = []
        self.completed_contributions = 0
        self.prediction_shape = None
        self.prediction_bytes = None

    def record(self, component, elapsed, phase=None):
        self.elapsed[(phase or self.phase, component)].append(float(elapsed))

    @contextmanager
    def use_phase(self, phase):
        old_phase = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = old_phase

    def rows(self):
        rows = []
        for (phase, component), values in sorted(self.elapsed.items()):
            rows.append({
                'phase': phase,
                'component': component,
                'calls': len(values),
                'total_seconds': float(np.sum(values)),
                'mean_ms': float(1000 * np.mean(values)),
                'p50_ms': float(1000 * percentile(values, 50)),
                'p90_ms': float(1000 * percentile(values, 90)),
            })
        return rows

    def cache_summary(self):
        unique_score_keys = len(set(self.score_roster_keys))
        total_score_keys = len(self.score_roster_keys)
        buy_score_keys = self.score_roster_keys[::2]
        pass_score_keys = self.score_roster_keys[1::2]
        unique_buy_score_keys = len(set(buy_score_keys))
        unique_pass_score_keys = len(set(pass_score_keys))
        return {
            'contribution_calls': self.contribution_calls,
            'candidate_evaluations': self.candidate_evaluations,
            'mean_selected_candidates': float(np.mean(self.selected_candidates)),
            'mean_unique_contexts_per_trial': float(np.mean(self.unique_contexts)),
            'context_cache_checks': self.context_cache_checks,
            'context_cache_hits': self.context_cache_hits,
            'context_cache_misses': self.context_cache_misses,
            'context_cache_hit_rate': float(
                self.context_cache_hits / max(self.context_cache_checks, 1)
            ),
            'score_calls': total_score_keys,
            'unique_trial_roster_scores': unique_score_keys,
            'reusable_score_calls': total_score_keys - unique_score_keys,
            'reusable_score_rate': float(
                1 - unique_score_keys / max(total_score_keys, 1)
            ),
            'buy_score_calls': len(buy_score_keys),
            'unique_buy_roster_scores': unique_buy_score_keys,
            'reusable_buy_score_rate': float(
                1 - unique_buy_score_keys / max(len(buy_score_keys), 1)
            ),
            'pass_score_calls': len(pass_score_keys),
            'unique_pass_roster_scores': unique_pass_score_keys,
            'reusable_pass_score_rate': float(
                1 - unique_pass_score_keys / max(len(pass_score_keys), 1)
            ),
            'completed_candidate_contributions': self.completed_contributions,
            'prediction_shape': self.prediction_shape,
            'prediction_bytes': self.prediction_bytes,
            'candidate_dataframe_copy_bytes': (
                None
                if self.prediction_bytes is None
                else self.prediction_bytes * self.candidate_evaluations
            ),
            'solve_status_counts': {
                '|'.join(key): value
                for key, value in sorted(self.status_counts.items())
            },
        }


@contextmanager
def instrument_target(instrumentation):
    cls = zsim.FootballSimulation
    original_matrix = zsim.matrix
    original_dataframe_copy = pd.DataFrame.copy

    original_solve_descriptor = cls.__dict__['solve_ilp']
    original_solve = cls.solve_ilp
    original_normalize_descriptor = cls.__dict__['normalize_salary_market']
    original_normalize = cls.normalize_salary_market
    original_normalize_values_descriptor = cls.__dict__['normalize_salary_market_values']
    original_normalize_values = cls.normalize_salary_market_values
    original_salary_descriptor = cls.__dict__['create_G_salaries']
    original_salary = cls.create_G_salaries
    original_salary_values_descriptor = cls.__dict__['create_G_salaries_from_values']
    original_salary_values = cls.create_G_salaries_from_values
    original_decision_descriptor = cls.__dict__['build_managed_decision_scores']
    original_decision = cls.build_managed_decision_scores
    original_marginal_descriptor = cls.__dict__['managed_marginal_values']
    original_marginal = cls.managed_marginal_values
    original_marginal_batch_descriptor = cls.__dict__[
        'managed_marginal_values_batch'
    ]
    original_marginal_batch = cls.managed_marginal_values_batch
    original_base_descriptor = cls.__dict__['managed_base_lineup_state']
    original_base = cls.managed_base_lineup_state
    original_multi_descriptor = cls.__dict__['managed_lineup_multi_context_scores']
    original_multi = cls.managed_lineup_multi_context_scores

    original_template = cls.sample_template_weekly_scores
    original_seeded_template = cls.sample_seeded_template_weekly_contexts
    original_value_matrix = cls.sample_managed_value_matrix
    original_holdout = cls.sample_managed_holdout_contexts
    original_contribution = cls.managed_roster_buy_pass_contributions
    original_static = cls.build_managed_ilp_static_matrices
    original_get_predictions = cls.get_predictions
    original_get_salaries = cls.get_salaries
    original_summary = cls.summarize_managed_roster_iteration
    original_tally_descriptor = cls.__dict__['tally_player_selections']
    original_tally = cls.tally_player_selections
    original_final = cls.final_results

    def timed_call(component, callback, *args, **kwargs):
        phase = instrumentation.phase
        start = time.perf_counter()
        try:
            return callback(*args, **kwargs)
        finally:
            instrumentation.record(
                component,
                time.perf_counter() - start,
                phase=phase,
            )

    def timed_matrix(*args, **kwargs):
        return timed_call('cvxopt_matrix_conversion', original_matrix, *args, **kwargs)

    def timed_dataframe_copy(frame, *args, **kwargs):
        return timed_call(
            'dataframe_copy',
            original_dataframe_copy,
            frame,
            *args,
            **kwargs,
        )

    def timed_solve(c, g, h, a, b):
        phase = instrumentation.phase
        if phase == 'contribution':
            branch = (
                'glpk_buy'
                if instrumentation.contribution_solve_ordinal % 2 == 0
                else 'glpk_pass'
            )
            instrumentation.contribution_solve_ordinal += 1
        else:
            branch = 'glpk_outer'
        start = time.perf_counter()
        result = original_solve(c, g, h, a, b)
        instrumentation.record(branch, time.perf_counter() - start, phase=phase)
        instrumentation.status_counts[(phase, branch, result[0])] += 1
        return result

    def timed_normalize(*args, **kwargs):
        return timed_call('salary_normalization', original_normalize, *args, **kwargs)

    def timed_normalize_values(*args, **kwargs):
        return timed_call(
            'salary_value_normalization',
            original_normalize_values,
            *args,
            **kwargs,
        )

    def timed_salary(*args, **kwargs):
        return timed_call('salary_constraint_row', original_salary, *args, **kwargs)

    def timed_salary_values(*args, **kwargs):
        return timed_call(
            'salary_value_constraint_row',
            original_salary_values,
            *args,
            **kwargs,
        )

    def timed_decision(*args, **kwargs):
        return timed_call('decision_scores', original_decision, *args, **kwargs)

    def timed_marginal(class_arg, *args, **kwargs):
        return timed_call('managed_marginal_values', original_marginal, *args, **kwargs)

    def timed_marginal_batch(class_arg, *args, **kwargs):
        return timed_call(
            'managed_marginal_values_batch',
            original_marginal_batch,
            *args,
            **kwargs,
        )

    def timed_base(class_arg, *args, **kwargs):
        return timed_call('managed_base_lineup_state', original_base, *args, **kwargs)

    def timed_multi(class_arg, *args, **kwargs):
        if instrumentation.phase == 'contribution':
            player_names = np.asarray(args[3], dtype=object)
            instrumentation.score_roster_keys.append((
                instrumentation.current_contribution,
                tuple(sorted(str(player) for player in player_names)),
            ))
        return timed_call('multi_context_roster_score', original_multi, *args, **kwargs)

    def timed_template(self, *args, **kwargs):
        return timed_call('sample_weekly_template', original_template, self, *args, **kwargs)

    def timed_seeded_template(self, *args, **kwargs):
        return timed_call(
            'sample_seeded_weekly_templates',
            original_seeded_template,
            self,
            *args,
            **kwargs,
        )

    def timed_value_matrix(self, *args, **kwargs):
        start = time.perf_counter()
        with instrumentation.use_phase('context_bank'):
            result = original_value_matrix(self, *args, **kwargs)
        instrumentation.record(
            'context_bank_total',
            time.perf_counter() - start,
            phase='context_bank',
        )
        return result

    def timed_holdout(self, *args, **kwargs):
        start = time.perf_counter()
        with instrumentation.use_phase('holdout'):
            result = original_holdout(self, *args, **kwargs)
        instrumentation.record(
            'holdout_total',
            time.perf_counter() - start,
            phase='holdout',
        )
        return result

    def timed_contribution(self, *args, **kwargs):
        predictions = args[0]
        selected_mask = np.asarray(args[1], dtype=bool)
        managed_context_indices = np.asarray(args[4], dtype=int)
        buy_value_cache = args[5]
        fixed_players = list(kwargs.get('fixed_players') or [])
        players = predictions.player.to_numpy()
        fixed_mask = np.isin(players, fixed_players)
        candidate_indices = np.flatnonzero(selected_mask & ~fixed_mask)
        context_indices = np.unique(managed_context_indices)

        if instrumentation.prediction_shape is None:
            instrumentation.prediction_shape = list(predictions.shape)
            instrumentation.prediction_bytes = int(
                predictions.memory_usage(deep=True).sum()
            )

        instrumentation.contribution_calls += 1
        instrumentation.current_contribution = instrumentation.contribution_calls - 1
        instrumentation.contribution_solve_ordinal = 0
        instrumentation.candidate_evaluations += len(candidate_indices)
        instrumentation.selected_candidates.append(len(candidate_indices))
        instrumentation.unique_contexts.append(len(context_indices))
        for candidate_idx in candidate_indices:
            candidate = players[candidate_idx]
            cache = buy_value_cache.get(candidate)
            for context_idx in context_indices:
                instrumentation.context_cache_checks += 1
                is_hit = bool(
                    cache is not None
                    and cache['computed'][context_idx]
                )
                if is_hit:
                    instrumentation.context_cache_hits += 1
                else:
                    instrumentation.context_cache_misses += 1

        start = time.perf_counter()
        with instrumentation.use_phase('contribution'):
            result = original_contribution(self, *args, **kwargs)
        instrumentation.record(
            'contribution_total',
            time.perf_counter() - start,
            phase='contribution',
        )
        instrumentation.completed_contributions += len(result)
        instrumentation.current_contribution = None
        return result

    def timed_static(self, *args, **kwargs):
        return timed_call('build_static_matrices', original_static, self, *args, **kwargs)

    def timed_get_predictions(self, *args, **kwargs):
        return timed_call('sample_projection_matrix', original_get_predictions, self, *args, **kwargs)

    def timed_get_salaries(self, *args, **kwargs):
        return timed_call('sample_salary_matrix', original_get_salaries, self, *args, **kwargs)

    def timed_summary(self, *args, **kwargs):
        return timed_call('managed_iteration_summary', original_summary, self, *args, **kwargs)

    def timed_tally(*args, **kwargs):
        return timed_call('tally_selections', original_tally, *args, **kwargs)

    def timed_final(self, *args, **kwargs):
        return timed_call('final_results', original_final, self, *args, **kwargs)

    zsim.matrix = timed_matrix
    pd.DataFrame.copy = timed_dataframe_copy
    cls.solve_ilp = staticmethod(timed_solve)
    cls.normalize_salary_market = staticmethod(timed_normalize)
    cls.normalize_salary_market_values = staticmethod(timed_normalize_values)
    cls.create_G_salaries = staticmethod(timed_salary)
    cls.create_G_salaries_from_values = staticmethod(timed_salary_values)
    cls.build_managed_decision_scores = staticmethod(timed_decision)
    cls.managed_marginal_values = classmethod(timed_marginal)
    cls.managed_marginal_values_batch = classmethod(timed_marginal_batch)
    cls.managed_base_lineup_state = classmethod(timed_base)
    cls.managed_lineup_multi_context_scores = classmethod(timed_multi)
    cls.sample_template_weekly_scores = timed_template
    cls.sample_seeded_template_weekly_contexts = timed_seeded_template
    cls.sample_managed_value_matrix = timed_value_matrix
    cls.sample_managed_holdout_contexts = timed_holdout
    cls.managed_roster_buy_pass_contributions = timed_contribution
    cls.build_managed_ilp_static_matrices = timed_static
    cls.get_predictions = timed_get_predictions
    cls.get_salaries = timed_get_salaries
    cls.summarize_managed_roster_iteration = timed_summary
    cls.tally_player_selections = staticmethod(timed_tally)
    cls.final_results = timed_final
    try:
        yield
    finally:
        zsim.matrix = original_matrix
        pd.DataFrame.copy = original_dataframe_copy
        cls.solve_ilp = original_solve_descriptor
        cls.normalize_salary_market = original_normalize_descriptor
        cls.normalize_salary_market_values = original_normalize_values_descriptor
        cls.create_G_salaries = original_salary_descriptor
        cls.create_G_salaries_from_values = original_salary_values_descriptor
        cls.build_managed_decision_scores = original_decision_descriptor
        cls.managed_marginal_values = original_marginal_descriptor
        cls.managed_marginal_values_batch = original_marginal_batch_descriptor
        cls.managed_base_lineup_state = original_base_descriptor
        cls.managed_lineup_multi_context_scores = original_multi_descriptor
        cls.sample_template_weekly_scores = original_template
        cls.sample_seeded_template_weekly_contexts = original_seeded_template
        cls.sample_managed_value_matrix = original_value_matrix
        cls.sample_managed_holdout_contexts = original_holdout
        cls.managed_roster_buy_pass_contributions = original_contribution
        cls.build_managed_ilp_static_matrices = original_static
        cls.get_predictions = original_get_predictions
        cls.get_salaries = original_get_salaries
        cls.summarize_managed_roster_iteration = original_summary
        cls.tally_player_selections = original_tally_descriptor
        cls.final_results = original_final


def compare_results(baseline, instrumented):
    baseline = baseline.reset_index(drop=True)
    instrumented = instrumented.reset_index(drop=True)
    if list(baseline.columns) != list(instrumented.columns):
        return {'columns_equal': False, 'values_equal': False}
    numeric_columns = baseline.select_dtypes(include=[np.number]).columns
    nonnumeric_columns = [
        column for column in baseline.columns if column not in numeric_columns
    ]
    max_difference = float(np.nanmax(np.abs(
        baseline[numeric_columns].to_numpy(dtype=float)
        - instrumented[numeric_columns].to_numpy(dtype=float)
    )))
    nonnumeric_equal = all(
        baseline[column].fillna('<NA>').equals(
            instrumented[column].fillna('<NA>')
        )
        for column in nonnumeric_columns
    )
    return {
        'columns_equal': True,
        'max_numeric_difference': max_difference,
        'nonnumeric_equal': nonnumeric_equal,
        'values_equal': bool(max_difference == 0 and nonnumeric_equal),
    }


def run_cprofile(iterations, seed=SEED):
    profiler = cProfile.Profile()
    profiler.enable()
    result, summary = run_serial(iterations, seed=seed)
    profiler.disable()
    profiler.dump_stats(RESULTS_DIR / 'target_serial.prof')
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative')
    stats.print_stats(120)
    (RESULTS_DIR / 'target_serial_cprofile.txt').write_text(
        stream.getvalue(),
        encoding='utf-8',
    )
    return result, summary


def run_parallel(iterations, max_workers, seed=SEED):
    conn, sim = create_sim()
    try:
        market = load_market_state(conn)
        kwargs = target_kwargs(sim, market)
        start = time.perf_counter()
        result = sim.run_sim_parallel(
            market['to_add'],
            market['to_drop'],
            iterations,
            max_workers=max_workers,
            block_size=50,
            random_seed=seed,
            **kwargs,
        )
        elapsed = time.perf_counter() - start
        summary = summarize_result(result, sim, elapsed, iterations)
        summary.update({
            'workers': int(sim.parallel_workers_used),
            'blocks': int(sim.parallel_blocks),
        })
        return result, summary
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-iters', type=int, default=50)
    parser.add_argument('--cprofile-iters', type=int, default=25)
    parser.add_argument(
        '--parallel-iters',
        type=int,
        nargs='*',
        default=[50, 100, 200, 400, 800, 1000],
    )
    parser.add_argument('--parallel-repeats', type=int, default=1)
    parser.add_argument('--max-workers', type=int, default=16)
    parser.add_argument('--skip-cprofile', action='store_true')
    parser.add_argument('--skip-parallel', action='store_true')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_result, baseline_summary = run_serial(args.serial_iters)
    print(
        f"serial baseline: {baseline_summary['elapsed_seconds']:.3f}s",
        flush=True,
    )

    instrumentation = TargetInstrumentation()
    instrumented_result, instrumented_summary = run_serial(
        args.serial_iters,
        instrumentation=instrumentation,
    )
    equivalence = compare_results(baseline_result, instrumented_result)
    if not equivalence['values_equal']:
        raise AssertionError(f'Instrumentation changed Target outputs: {equivalence}')
    print(
        f"serial instrumented: {instrumented_summary['elapsed_seconds']:.3f}s",
        flush=True,
    )

    pd.DataFrame(instrumentation.rows()).to_csv(
        RESULTS_DIR / 'component_timings.csv',
        index=False,
    )
    cache_summary = instrumentation.cache_summary()
    (RESULTS_DIR / 'cache_metrics.json').write_text(
        json.dumps(cache_summary, indent=2),
        encoding='utf-8',
    )
    serial_output = {
        'baseline': baseline_summary,
        'instrumented': instrumented_summary,
        'equivalence': equivalence,
    }
    (RESULTS_DIR / 'serial_summary.json').write_text(
        json.dumps(serial_output, indent=2),
        encoding='utf-8',
    )

    if not args.skip_cprofile:
        _, cprofile_summary = run_cprofile(args.cprofile_iters)
        (RESULTS_DIR / 'cprofile_summary.json').write_text(
            json.dumps(cprofile_summary, indent=2),
            encoding='utf-8',
        )
        print(
            f"cProfile run: {cprofile_summary['elapsed_seconds']:.3f}s",
            flush=True,
        )

    scaling_rows = []
    if not args.skip_parallel:
        for iterations in args.parallel_iters:
            for repeat in range(max(1, int(args.parallel_repeats))):
                _, summary = run_parallel(
                    iterations,
                    args.max_workers,
                    seed=SEED + repeat,
                )
                summary['repeat'] = repeat + 1
                scaling_rows.append(summary)
                print(
                    f"parallel {iterations} trials, "
                    f"{summary['workers']} workers: "
                    f"{summary['elapsed_seconds']:.3f}s",
                    flush=True,
                )

        baseline_block_seconds = baseline_summary['elapsed_seconds']
        for row in scaling_rows:
            serial_equivalent = baseline_block_seconds * math.ceil(
                row['iterations'] / args.serial_iters
            )
            row['serial_equivalent_seconds'] = serial_equivalent
            row['parallel_speedup'] = serial_equivalent / row['elapsed_seconds']
            row['worker_efficiency'] = (
                row['parallel_speedup'] / max(row['workers'], 1)
            )
        pd.DataFrame(scaling_rows).to_csv(
            RESULTS_DIR / 'parallel_scaling.csv',
            index=False,
        )

    output = {
        'configuration': {
            'year': YEAR,
            'league': LEAGUE,
            'salary_cap': SALARY_CAP,
            'roster_size': ROSTER_SIZE,
            'serial_iterations': args.serial_iters,
            'cprofile_iterations': args.cprofile_iters,
            'max_workers': args.max_workers,
        },
        'serial': serial_output,
        'cache': cache_summary,
        'parallel': scaling_rows,
    }
    (RESULTS_DIR / 'profile_summary.json').write_text(
        json.dumps(output, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
