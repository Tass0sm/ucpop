import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import unified_planning as up
from unified_planning.shortcuts import OneshotPlanner

from problems import PROBLEM_BUILDERS

SOLVERS = [
    'pop',
    'pop2',
    'pcop',
    'ucpop',
    'fast-downward',
]

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def is_solved(status) -> bool:
    return status in {
        up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING,
        up.engines.PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    }


def plan_kind(plan) -> str:
    if plan is None:
        return ''
    return type(plan).__name__


def plan_size(plan):
    if plan is None:
        return ''
    if hasattr(plan, 'actions'):
        try:
            return len(plan.actions)
        except TypeError:
            pass
    return ''

def run_one(problem_name: str, builder, solver_name: str) -> dict:
    problem = builder()
    start = time.perf_counter()

    try:
        with OneshotPlanner(name=solver_name) as planner:
            result = planner.solve(problem)
        elapsed = time.perf_counter() - start

        return {
            'problem': problem_name,
            'solver': solver_name,
            'status': str(result.status),
            'solved': is_solved(result.status),
            'time_sec': round(elapsed, 6),
            'plan_kind': plan_kind(result.plan),
            'plan_size': plan_size(result.plan),
            'error': '',
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            'problem': problem_name,
            'solver': solver_name,
            'status': 'ERROR',
            'solved': False,
            'time_sec': round(elapsed, 6),
            'plan_kind': '',
            'plan_size': '',
            'error': str(exc),
        }

def write_csv(rows) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = REPORTS_DIR / f'benchmark_{timestamp}.csv'
    fieldnames = [
        'problem',
        'solver',
        'status',
        'solved',
        'time_sec',
        'plan_kind',
        'plan_size',
        'error',
    ]

    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def format_cell(row: dict) -> str:
    if row['solved']:
        return f"OK ({row['time_sec']:.4f}s)"
    if row['status'] == 'ERROR':
        return 'ERROR'
    return str(row['status'])


def print_summary_table(rows) -> None:
    problems = list(PROBLEM_BUILDERS.keys())
    solvers = SOLVERS

    row_map = {(row['problem'], row['solver']): row for row in rows}

    headers = ['problem'] + solvers
    table = [headers]
    for problem in problems:
        current = [problem]
        for solver in solvers:
            current.append(format_cell(row_map[(problem, solver)]))
        table.append(current)

    widths = [max(len(str(r[i])) for r in table) for i in range(len(headers))]

    for idx, row in enumerate(table):
        line = ' | '.join(str(value).ljust(widths[i]) for i, value in enumerate(row))
        print(line)
        if idx == 0:
            print('-+-'.join('-' * width for width in widths))

def main():
    rows = []
    for problem_name, builder in PROBLEM_BUILDERS.items():
        for solver_name in SOLVERS:
            logging.info('Running %s on %s', solver_name, problem_name)
            row = run_one(problem_name, builder, solver_name)
            rows.append(row)

    print('\nBenchmark summary:\n')
    print_summary_table(rows)

    csv_path = write_csv(rows)
    print(f'\nWrote CSV report to: {csv_path}')

if __name__ == '__main__':
    main()
