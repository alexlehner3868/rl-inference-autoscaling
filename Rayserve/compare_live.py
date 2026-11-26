"""Compare completed fixed/queue/DQN/PPO Ray Serve live runs."""

import argparse
import json
from pathlib import Path
import csv

root = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='*', type=Path, help='Run folders containing summary.json')
    args = parser.parse_args()

    folders = args.folders or sorted((root / 'results/rayserve_live').glob('*/*'))
    rows = []
    for folder in folders:
        file = folder / 'summary.json'
        if file.is_file():
            rows.append(json.loads(file.read_text()))

    if not rows:
        raise SystemExit('No Ray Serve summaries found. Run live_autoscale.py first.')

    print('Controller Workload Seconds  On-time %  Avg replicas  P95 (s)  Scales  Drops')

    # Save the comparison results
    output_file = root / 'results/rayserve_live/comparison.csv'

    with open(output_file, 'w', newline='') as f:
        columns = ['controller', 'workload', 'duration_seconds', 'on_time_percentage', 'average_workers', 'p95_latency_seconds', 'scaling_changes', 'drops']

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            result = {
                'controller': row['controller'],
                'workload': row['workload'],
                'duration_seconds': row['duration_seconds'],
                'on_time_percentage': row['on_time_percentage'],
                'average_workers': row['average_workers'],
                'p95_latency_seconds': row['p95_latency_seconds'],
                'scaling_changes': row['scaling_changes'],
                'drops': row['dropped_full'] + row['dropped_deadline']}

            writer.writerow(result)

    # Check whether the experiments used the same workload and duration
    workloads = {(row['workload'], row['duration_seconds']) for row in rows}

    if len(workloads) > 1:
        warning = 'WARNING: Experiments used different workloads or durations.'
        print(warning)

        with open(root / 'results/rayserve_live/comparison_warning.txt', 'w') as f:
            f.write(warning)

    print('Results saved in:', output_file)



if __name__ == '__main__':
    main()
