import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import unified_planning as up
from unified_planning.shortcuts import OneshotPlanner, get_environment

import ucpop  # noqa: F401 - ensures custom planners register
from stress_metrics import extract_plan_metrics, is_solved, plan_kind
from stress_registry import PCOP_STRESS_PROBLEMS

# Keep solver logs quiet during benchmarking.
ucpop.search.logger.setLevel(logging.ERROR)
ucpop.pop.logger.setLevel(logging.ERROR)
ucpop.pop2.logger.setLevel(logging.ERROR)
ucpop.classes.logger.setLevel(logging.ERROR)
get_environment().credits_stream = None

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SOLVER = 'pcop'


def run_one(problem_name: str, spec: dict, solver_name: str, timeout_s: float | None) -> dict:
    builder = spec['builder']
    problem = builder()
    start = time.perf_counter()

    try:
        with OneshotPlanner(name=solver_name) as planner:
            try:
                result = planner.solve(problem, timeout=timeout_s)
            except TypeError:
                result = planner.solve(problem)
        elapsed = time.perf_counter() - start

        metrics = extract_plan_metrics(
            result.plan,
            sync_actions=spec.get('sync_actions', set()),
            resource_actions=spec.get('resource_actions', set()),
        )

        return {
            'problem': problem_name,
            'family': spec['family'],
            'solver': solver_name,
            'status': str(result.status),
            'solved': is_solved(result.status),
            'time_sec': round(elapsed, 6),
            'plan_kind': plan_kind(result.plan),
            'summary': spec.get('summary', ''),
            'error': '',
            **metrics,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            'problem': problem_name,
            'family': spec['family'],
            'solver': solver_name,
            'status': 'ERROR',
            'solved': False,
            'time_sec': round(elapsed, 6),
            'plan_kind': '',
            'summary': spec.get('summary', ''),
            'error': str(exc),
            'action_count': '',
            'direct_orderings': '',
            'comparable_pairs': '',
            'unordered_pairs': '',
            'flexibility_ratio': '',
            'unique_agent_count': '',
            'sync_action_count': '',
            'resource_action_count': '',
            'binding_var_count': '',
            'possible_binding_count': '',
            'invalid_binding_count': '',
            'action_histogram': '',
        }


def write_csv(rows, stem: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = REPORTS_DIR / f'{stem}_{timestamp}.csv'
    fieldnames = [
        'problem',
        'family',
        'solver',
        'status',
        'solved',
        'time_sec',
        'plan_kind',
        'action_count',
        'direct_orderings',
        'comparable_pairs',
        'unordered_pairs',
        'flexibility_ratio',
        'unique_agent_count',
        'sync_action_count',
        'resource_action_count',
        'binding_var_count',
        'possible_binding_count',
        'invalid_binding_count',
        'action_histogram',
        'summary',
        'error',
    ]

    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def format_cell(row: dict) -> str:
    if row['solved']:
        return f"OK ({row['time_sec']:.3f}s, A={row['action_count']}, F={row['flexibility_ratio']})"
    if row['status'] == 'ERROR':
        return 'ERROR'
    return str(row['status'])

def print_summary_table(rows) -> None:
    headers = ['problem', 'family', 'result']
    table = [headers]

    for row in rows:
        table.append([row['problem'], row['family'], format_cell(row)])

    widths = [max(len(str(r[i])) for r in table) for i in range(len(headers))]
    for idx, row in enumerate(table):
        line = ' | '.join(str(v).ljust(widths[i]) for i, v in enumerate(row))
        print(line)
        if idx == 0:
            print('-+-'.join('-' * w for w in widths))


def parse_args():
    parser = argparse.ArgumentParser(description='Run PCOP-focused stress benchmarks.')
    parser.add_argument('--solver', default=DEFAULT_SOLVER, help='Planner name to use. Defaults to pcop.')
    parser.add_argument('--timeout', type=float, default=None, help='Optional per-problem timeout in seconds.')
    parser.add_argument('--only', nargs='*', default=None, help='Optional subset of problem names to run.')
    parser.add_argument('--stem', default='pcop_stress', help='CSV filename stem.')
    return parser.parse_args()


def main():
    args = parse_args()
    selected = PCOP_STRESS_PROBLEMS
    if args.only:
        selected = {k: v for k, v in PCOP_STRESS_PROBLEMS.items() if k in set(args.only)}

    rows = []
    for problem_name, spec in selected.items():
        logging.info('Running %s on %s', args.solver, problem_name)
        rows.append(run_one(problem_name, spec, args.solver, args.timeout))

    print('\nPCOP stress benchmark summary:\n')
    print_summary_table(rows)
    csv_path = write_csv(rows, args.stem)
    print(f'\nWrote CSV report to: {csv_path}')


if __name__ == '__main__':
    main()
