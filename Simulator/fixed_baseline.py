import json
import csv
from pathlib import Path
from datetime import datetime
from Simulator.Simulator_queue import Simulator, SimulatorConfig

# Settings - change these if needed
splits = ['train', 'validation', 'test']
worker_counts = [1, 2, 4]
deadline = 1.0
queue_size = 50

root = Path(__file__).resolve().parent
calibration_file = root / 'results/calibration/simulator_parameters.json'
output_folder = root / 'results/fixed_baseline' / datetime.now().strftime('%Y%m%d_%H%M%S')

# Read measured YOLO performance
with open(calibration_file) as f:
    calibration = json.load(f)

service_time = calibration['service_time_mean_seconds']
startup_time = calibration['startup_mean_seconds']

all_results = []

# Go through each workload file in each split
for split in splits:
    workload_folder = root / 'workloads' / split

    for file in sorted(workload_folder.glob('*.json')):
        with open(file) as f:
            workload = json.load(f)

        # Test the same workload with different fixed worker counts
        for workers in worker_counts:
            config = SimulatorConfig(
                camera_fps=workload['camera_fps'],
                service_seconds=service_time,
                startup_seconds=startup_time,
                deadline_seconds=deadline,
                decision_seconds=workload['decision_seconds'],
                initial_workers=workers,
                max_workers=workers,
                queue_capacity_frames=queue_size
            )
            sim = Simulator(config)

            # One hold action for each decision period: no autoscaling
            time = 0
            for interval in workload['intervals']:
                if abs(time - interval['start_seconds']) > 0.000001:
                    raise ValueError('Workload has a gap or overlapping intervals: ' + file.name)

                while time < interval['end_seconds'] - 0.000001:
                    sim.step('hold', interval['camera_ids'])
                    time = round(time + config.decision_seconds, 9)

            if abs(time - workload['duration_seconds']) > 0.000001:
                raise ValueError('Workload duration does not match: ' + file.name)

            # Check that no frames went missing
            waiting = len(sim.queue)
            processing = 0
            for worker in sim.workers:
                if worker['frame'] is not None:
                    processing += 1

            counted = sim.on_time + sim.late + sim.dropped_deadline + sim.dropped_full + waiting + processing
            if sim.arrived != counted:
                raise ValueError('Frame counts do not match: ' + file.name)

            # Save this run's report, history and demand/capacity graph
            folder_name = file.stem + '_' + str(workers) + '_workers'
            run_folder = output_folder / split / folder_name
            sim.save_results(run_folder, controller='fixed_' + str(workers), workload=file.stem)

            # Read saved report and collect one row for the comparison CSV
            with open(run_folder / 'summary.json') as f:
                summary = json.load(f)

            result = {
                'split': split,
                'workload': file.stem,
                'pattern': workload['pattern'],
                'load_level': workload['load_level'],
                'workers': workers,
                'arrived': summary['arrived'],
                'on_time': summary['on_time'],
                'late': summary['late'],
                'dropped_deadline': summary['dropped_deadline'],
                'dropped_full': summary['dropped_full'],
                'unfinished_frames': summary['unfinished_frames'],
                'on_time_percentage': summary['on_time_percentage'],
                'worker_seconds': summary['worker_seconds'],
                'p95_latency_seconds': summary['completed_latency_seconds']['p95'],
                'max_queue_length': summary['max_queue_length'],
                'scaling_changes': summary['scaling_changes'],
                'result_dir': str(run_folder.relative_to(root))
            }
            all_results.append(result)
            print(split, file.stem, workers, 'workers:', round(result['on_time_percentage'], 2) if result['on_time_percentage'] is not None else 0, '% on time')

# Save a single CSV for comparing all runs
if all_results:
    with open(output_folder / 'comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print('Finished', len(all_results), 'runs')
    print('Results saved in:', output_folder)
else:
    print('No workload files were found.')
