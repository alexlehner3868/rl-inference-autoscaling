import json
import csv
from pathlib import Path
from datetime import datetime
from Simulator_queue import Simulator, SimulatorConfig

# Settings (change these when experimenting)
splits = ['train', 'validation']
add_threshold = 5       # Add a worker if 5 or more frames are waiting
remove_threshold = 1    # Consider removing a worker if 1 or fewer are waiting
max_workers = 4
queue_size = 50
deadline = 1.0

root = Path(__file__).resolve().parent.parent

calibration_file = root / 'results/calibration/simulator_parameters.json'
output_folder = root / 'results/queue_threshold' / datetime.now().strftime('%Y%m%d_%H%M%S')

with open(calibration_file) as f:
    calibration = json.load(f)

service_time = calibration['service_time_mean_seconds']
startup_time = calibration['startup_mean_seconds']
all_results = []

for split in splits:
    workload_folder = root / 'workloads' / split

    for file in sorted(workload_folder.glob('*.json')):
        with open(file) as f:
            workload = json.load(f)

        config = SimulatorConfig(
            camera_fps=workload['camera_fps'],
            service_seconds=service_time,
            startup_seconds=startup_time,
            deadline_seconds=deadline,
            decision_seconds=workload['decision_seconds'],
            initial_workers=1,
            max_workers=max_workers,
            queue_capacity_frames=queue_size)
        sim = Simulator(config)
        time = 0

        for interval in workload['intervals']:
            if abs(time - interval['start_seconds']) > 0.000001:
                raise ValueError('Invalid workload intervals: ' + file.name)

            while time < interval['end_seconds'] - 0.000001:
                cameras = interval['camera_ids']
                waiting = len(sim.queue)
                workers = len(sim.workers)
                action = 'hold'
                # When the queue builds up, start an extra worker.
                if waiting >= add_threshold and workers < max_workers:
                    action = 'add'
                # Remove a worker only if the remaining workers can handle the currently active cameras without building a new queue.
                elif waiting <= remove_threshold and workers > 1:
                    all_ready = True
                    for worker in sim.workers:
                        if worker['ready_at'] > sim.tick or worker['retiring']:
                            all_ready = False

                    demand_fps = len(cameras) * config.camera_fps
                    capacity_after_remove = (workers - 1) / config.service_seconds
                    if all_ready and demand_fps <= capacity_after_remove:
                        action = 'remove'

                sim.step(action,cameras)
                time = round(time + config.decision_seconds, 9)

        if abs(time - workload['duration_seconds']) > 0.000001:
            raise ValueError('Workload duration mismatch: ' + file.name)

        
        waiting = len(sim.queue)
        processing = 0
        for worker in sim.workers:
            if worker['frame'] is not None:
                processing += 1

        counted = (sim.on_time + sim.late + sim.dropped_deadline + sim.dropped_full + waiting + processing)
        if counted != sim.arrived:
            raise ValueError('Frame accounting mismatch: ' + file.name)

        folder = output_folder / split / file.stem
        sim.save_results(folder, controller='queue_threshold', workload=file.stem)

        with open(folder / 'summary.json') as f:
            summary = json.load(f)

        result = {
            'split': split,
            'workload': file.stem,
            'pattern': workload['pattern'],
            'load_level': workload['load_level'],
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
            'add_actions': summary['action_counts']['add'],
            'remove_actions': summary['action_counts']['remove'],
            'blocked_actions': summary['action_counts']['blocked'],
            'add_threshold': add_threshold,
            'remove_threshold': remove_threshold}
        all_results.append(result)
        print(split, file.stem, '- on time:', round(result['on_time_percentage'] or 0, 2), '%', '- worker-seconds:', round(result['worker_seconds'], 1), '- scaling changes:', result['scaling_changes'])

if all_results:
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / 'comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print('Finished', len(all_results), 'simulations')
    print('Results saved in:', output_folder)
else:
    print('No workload files found.')
