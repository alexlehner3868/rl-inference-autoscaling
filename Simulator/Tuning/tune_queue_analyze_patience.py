import json
import csv
from pathlib import Path
from datetime import datetime
from Simulator.Simulator_queue import Simulator, SimulatorConfig
import matplotlib.pyplot as plt

# Settings
splits = ['train', 'validation']
add_thresholds = [3, 5, 10]
remove_thresholds = [0, 2]
scale_down_patiences = [5, 10, 15, 20]
max_workers = 4
queue_size = 50
deadline = 1.0

root = Path(__file__).resolve().parent.parent.parent

# Set to False if you want to run the simulations again.
use_saved_results = False
saved_folder = root / 'results/queue_threshold_grid/20261110_202707'

if use_saved_results:
    output_folder = saved_folder
else:
    output_folder = root / 'results/queue_threshold_grid' / datetime.now().strftime('%Y%m%d_%H%M%S')


def summarize_results(all_results, split='validation'):
    groups = {}

    for row in all_results:
        if row['split'] != split:
            continue

        key = (row['add_threshold'], row['remove_threshold'], row['scale_down_patience'])
        groups.setdefault(key, []).append(row)

    summary = {}

    for key, rows in groups.items():
        arrived = sum(row['arrived'] for row in rows)
        on_time = sum(row['on_time'] for row in rows)
        worker_seconds = sum(row['worker_seconds'] for row in rows)
        scaling_changes = sum(row['scaling_changes'] for row in rows)
        add_actions = sum(row['add_actions'] for row in rows)
        remove_actions = sum(row['remove_actions'] for row in rows)
        blocked_actions = sum(row['blocked_actions'] for row in rows)
        duration = 300 * len(rows)

        summary[key] = {
            'on_time_percentage': 100 * on_time / arrived if arrived else 0,
            'average_workers': worker_seconds / duration,
            'scaling_changes': scaling_changes / len(rows),
            'add_actions': add_actions / len(rows),
            'remove_actions': remove_actions / len(rows),
            'blocked_actions': blocked_actions / len(rows)}

    return summary


# Figure 1: Patience versus deadline performance

def plot_patience_performance(summary, figures_folder):
    fig, ax = plt.subplots(figsize=(8, 5))

    for add in add_thresholds:
        points = sorted([(key[2], values) for key, values in summary.items() if key[0] == add and key[1] == 0])
        ax.plot([p[0] for p in points], [p[1]['on_time_percentage'] for p in points], marker='o', label=f'Add = {add}')

    ax.set_xlabel('Scale-down patience (decisions)')
    ax.set_ylabel('On-time frames (%)')
    ax.set_title('Validation: Patience vs Deadline Performance')
    ax.set_xticks(scale_down_patiences)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_folder / 'patience_performance.png', dpi=200)
    plt.close(fig)


# Figure 2: Patience versus resource usage

def plot_patience_resources(summary, figures_folder):
    fig, ax = plt.subplots(figsize=(8, 5))

    for add in add_thresholds:
        points = sorted([(key[2], values) for key, values in summary.items() if key[0] == add and key[1] == 0])
        ax.plot([p[0] for p in points], [p[1]['average_workers'] for p in points], marker='o', label=f'Add = {add}')

    ax.set_xlabel('Scale-down patience (decisions)')
    ax.set_ylabel('Average allocated workers')
    ax.set_title('Validation: Patience vs Resource Usage')
    ax.set_xticks(scale_down_patiences)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_folder / 'patience_resources.png', dpi=200)
    plt.close(fig)


# Figure 3: Patience versus scaling changes

def plot_patience_scaling(summary, figures_folder):
    fig, ax = plt.subplots(figsize=(8, 5))

    for add in add_thresholds:
        points = sorted([(key[2], values) for key, values in summary.items() if key[0] == add and key[1] == 0])
        ax.plot([p[0] for p in points], [p[1]['scaling_changes'] for p in points], marker='o', label=f'Add = {add}')

    ax.set_xlabel('Scale-down patience (decisions)')
    ax.set_ylabel('Average scaling changes per workload')
    ax.set_title('Validation: Patience vs Scaling Frequency')
    ax.set_xticks(scale_down_patiences)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_folder / 'patience_scaling.png', dpi=200)
    plt.close(fig)


all_results = []

if use_saved_results:
    # Read the results from the completed simulations.
    with open(output_folder / 'comparison.csv', newline='') as f:
        all_results = list(csv.DictReader(f))

    numbers = ['add_threshold', 'remove_threshold', 'scale_down_patience', 'arrived', 'on_time', 'worker_seconds', 'scaling_changes', 'add_actions', 'remove_actions', 'blocked_actions']
    for row in all_results:
        for key in numbers:
            row[key] = float(row[key])

else:
    # Run the grid search.
    calibration_file = root / 'results/calibration/simulator_parameters.json'
    with open(calibration_file) as f:
        calibration = json.load(f)

    service_time = calibration['service_time_mean_seconds']
    startup_time = calibration['startup_mean_seconds']

    for add_threshold in add_thresholds:
        for remove_threshold in remove_thresholds:
            if remove_threshold >= add_threshold:
                continue

            for scale_down_patience in scale_down_patiences:
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
                        low_queue_steps = 0

                        for interval in workload['intervals']:
                            if abs(time - interval['start_seconds']) > 0.000001:
                                raise ValueError('Invalid workload intervals: ' + file.name)

                            while time < interval['end_seconds'] - 0.000001:
                                cameras = interval['camera_ids']
                                waiting = len(sim.queue)
                                workers = len(sim.workers)
                                action = 'hold'

                                if waiting >= add_threshold and workers < max_workers:
                                    action = 'add'
                                    low_queue_steps = 0

                                elif waiting <= remove_threshold and workers > 1:
                                    all_ready = True
                                    for worker in sim.workers:
                                        if worker['ready_at'] > sim.tick or worker['retiring']:
                                            all_ready = False

                                    demand_fps = len(cameras) * config.camera_fps
                                    capacity_after_remove = (workers - 1) / config.service_seconds

                                    if all_ready and demand_fps <= capacity_after_remove:
                                        low_queue_steps += 1
                                        if low_queue_steps >= scale_down_patience:
                                            action = 'remove'
                                            low_queue_steps = 0
                                    else:
                                        low_queue_steps = 0
                                else:
                                    low_queue_steps = 0

                                sim.step(action, cameras)
                                time = round(time + config.decision_seconds, 9)

                        if abs(time - workload['duration_seconds']) > 0.000001:
                            raise ValueError('Workload duration mismatch: ' + file.name)

                        waiting = len(sim.queue)
                        processing = 0
                        for worker in sim.workers:
                            if worker['frame'] is not None:
                                processing += 1

                        counted = sim.on_time + sim.late + sim.dropped_deadline + sim.dropped_full + waiting + processing
                        if counted != sim.arrived:
                            raise ValueError('Frame accounting mismatch: ' + file.name)

                        folder = output_folder / f'add_{add_threshold}_remove_{remove_threshold}_patience_{scale_down_patience}' / split / file.stem
                        sim.save_results(folder, controller='queue_threshold', workload=file.stem)

                        with open(folder / 'summary.json') as f:
                            result_summary = json.load(f)

                        result = {
                            'split': split,
                            'workload': file.stem,
                            'pattern': workload['pattern'],
                            'load_level': workload['load_level'],
                            'arrived': result_summary['arrived'],
                            'on_time': result_summary['on_time'],
                            'late': result_summary['late'],
                            'dropped_deadline': result_summary['dropped_deadline'],
                            'dropped_full': result_summary['dropped_full'],
                            'unfinished_frames': result_summary['unfinished_frames'],
                            'on_time_percentage': result_summary['on_time_percentage'],
                            'worker_seconds': result_summary['worker_seconds'],
                            'p95_latency_seconds': result_summary['completed_latency_seconds']['p95'],
                            'max_queue_length': result_summary['max_queue_length'],
                            'scaling_changes': result_summary['scaling_changes'],
                            'add_actions': result_summary['action_counts']['add'],
                            'remove_actions': result_summary['action_counts']['remove'],
                            'blocked_actions': result_summary['action_counts']['blocked'],
                            'add_threshold': add_threshold,
                            'remove_threshold': remove_threshold,
                            'scale_down_patience': scale_down_patience
                        }
                        all_results.append(result)
                        print(split, file.stem, 'patience:', scale_down_patience, '- on time:', round(result['on_time_percentage'] or 0, 2), '%', '- worker-seconds:', round(result['worker_seconds'], 1), '- scaling changes:', result['scaling_changes'])

    if all_results:
        output_folder.mkdir(parents=True, exist_ok=True)
        with open(output_folder / 'comparison.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

if not all_results:
    raise SystemExit('No results found.')

print('Results:', len(all_results), 'simulations')
print('Results saved in:', output_folder)

summary = summarize_results(all_results)
figures_folder = output_folder / 'figures'
figures_folder.mkdir(parents=True, exist_ok=True)
plot_patience_performance(summary, figures_folder)
plot_patience_resources(summary, figures_folder)
plot_patience_scaling(summary, figures_folder)

print('Add | Remove | Patience | On-time % | Avg workers | Avg scaling | Avg adds | Avg removes | Avg blocked')
for (add, remove, patience), values in sorted(summary.items()):
    print(f'{int(add)}/{int(remove)} | {int(patience)} | {values["on_time_percentage"]:.2f} | {values["average_workers"]:.2f} | {values["scaling_changes"]:.1f} | {values["add_actions"]:.1f} | {values["remove_actions"]:.1f} | {values["blocked_actions"]:.1f}')

print('Figures saved in:', figures_folder)
