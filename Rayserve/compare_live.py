"""Compare one complete batch of Ray Serve validation runs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


root = Path(__file__).resolve().parent.parent
DEFAULT_CONTROLLERS = ('fixed', 'queue', 'predictive')
COLUMNS = (
    'controller', 'workload', 'duration_seconds', 'frame_deadline_seconds',
    'request_timeout_seconds', 'arrived', 'on_time', 'late', 'dropped_full',
    'dropped_deadline', 'failed', 'unfinished_frames', 'not_on_time_frames',
    'on_time_percentage', 'http_concurrency_limit',
    'worker_seconds', 'ready_worker_seconds',
    'average_workers', 'average_ready_workers', 'p95_latency_seconds',
    'scaling_changes',
)
MATCHING_SETTINGS = (
    'duration_seconds', 'scheduled_frames', 'frame_deadline_seconds',
    'request_timeout_seconds', 'queue_capacity_frames', 'max_workers', 'video',
)


def compare_batch(batch_dir, controllers=DEFAULT_CONTROLLERS):
    if not batch_dir.is_dir():
        raise ValueError(f'Batch directory does not exist: {batch_dir}')
    if len(controllers) != len(set(controllers)):
        raise ValueError('List each controller only once')

    validation_workloads = {path.stem for path in (root / 'workloads/validation').glob('*.json')}
    if len(validation_workloads) != 20:
        raise ValueError(f'Expected 20 validation workloads; found {len(validation_workloads)}')

    summary_paths = sorted(batch_dir.glob('*/summary.json'))
    if not summary_paths:
        raise ValueError(f'No run summaries found in {batch_dir}')

    rows = []
    required = (set(COLUMNS) - {'not_on_time_frames'}) | set(MATCHING_SETTINGS)
    for path in summary_paths:
        row = json.loads(path.read_text())
        missing_fields = required - row.keys()
        if missing_fields:
            raise ValueError(f'{path} is missing fields: {sorted(missing_fields)}')
        if row['arrived'] != row['scheduled_frames']:
            raise ValueError(f'{path} did not record every scheduled frame')
        accounted = sum(row[key] for key in (
            'on_time', 'late', 'dropped_full', 'dropped_deadline', 'failed', 'unfinished_frames'
        ))
        if row['arrived'] != accounted:
            raise ValueError(f'{path} has inconsistent frame counts')
        rows.append(row)

    expected = {(controller, workload) for controller in controllers for workload in validation_workloads}
    observed = Counter((row['controller'], row['workload']) for row in rows)
    missing = sorted(expected - observed.keys())
    unexpected = sorted(observed.keys() - expected)
    duplicates = sorted(key for key, count in observed.items() if count != 1)
    if missing or unexpected or duplicates:
        raise ValueError(
            f'Batch must contain one run per controller and validation workload; '
            f'missing={missing}, unexpected={unexpected}, duplicates={duplicates}'
        )

    for workload in validation_workloads:
        settings = {tuple(row[field] for field in MATCHING_SETTINGS) for row in rows if row['workload'] == workload}
        if len(settings) != 1:
            raise ValueError(f'Run settings differ across controllers for {workload}')

    output_file = batch_dir / 'comparison.csv'
    with output_file.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item['workload'], item['controller'])):
            result = {field: row[field] for field in COLUMNS if field != 'not_on_time_frames'}
            result['not_on_time_frames'] = row['arrived'] - row['on_time']
            writer.writerow(result)
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_dir', type=Path, help='Directory containing all run folders')
    parser.add_argument('--controllers', nargs='+', default=DEFAULT_CONTROLLERS)
    args = parser.parse_args()
    try:
        output_file = compare_batch(args.batch_dir, args.controllers)
    except ValueError as error:
        parser.error(str(error))
    print('Results saved in:', output_file)


if __name__ == '__main__':
    main()
