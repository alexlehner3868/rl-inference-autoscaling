import json
import csv
from pathlib import Path
from datetime import datetime
from  Simulator.Simulator_queue import Simulator, SimulatorConfig
import numpy as np
import matplotlib.pyplot as plt
# Settings (change these when experimenting)
splits = ['train', 'validation']
add_thresholds = [3, 5]
remove_thresholds = [0]
scale_down_patiences = 1
max_workers = 4
queue_size = 50
deadline = 1.0
  # Wait for 3 consecutive low-queue decisions
root = Path(__file__).resolve().parent.parent.parent

calibration_file = root / 'results/calibration/simulator_parameters.json'
output_folder = root / 'results/queue_threshold_grid' / datetime.now().strftime('%Y%m%d_%H%M%S')

with open(calibration_file) as f:
    calibration = json.load(f)

service_time = calibration['service_time_mean_seconds']
startup_time = calibration['startup_mean_seconds']
all_results = []

for add_threshold in add_thresholds:
    for remove_threshold in remove_thresholds:
        if remove_threshold >= add_threshold:
            continue

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

                        # Add a worker if the queue is too large.
                        if waiting >= add_threshold and workers < max_workers:
                            action = 'add'
                            low_queue_steps = 0

                        # Check whether we can safely remove a worker.
                        elif waiting <= remove_threshold and workers > 1:
                            all_ready = True

                            for worker in sim.workers:
                                if worker['ready_at'] > sim.tick or worker['retiring']:
                                    all_ready = False

                            demand_fps = len(cameras) * config.camera_fps
                            capacity_after_remove = (workers - 1) / config.service_seconds

                            if all_ready and demand_fps <= capacity_after_remove:
                                low_queue_steps += 1

                                # Remove only after 3 consecutive qualifying checks.
                                if low_queue_steps >= scale_down_patience:
                                    action = 'remove'
                                    low_queue_steps = 0
                            else:
                                low_queue_steps = 0

                        # Queue is between thresholds, or only one worker remains.
                        else:
                            low_queue_steps = 0

                        # Apply the action and advance the simulation by one second.
                        sim.step(action, cameras)
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

                folder = output_folder / f'add_{add_threshold}_remove_{remove_threshold}_patience_{scale_down_patience}' / split / file.stem
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
                    'remove_threshold': remove_threshold,
                    'scale_down_patience': scale_down_patience
                    }
                all_results.append(result)
                print(split, file.stem, '- on time:', round(result['on_time_percentage'] or 0, 2), '%', '- worker-seconds:', round(result['worker_seconds'], 1), '- scaling changes:', result['scaling_changes'])





# Calculate the results for each threshold configuration.
def summarize_results(all_results, split='validation', load_level=None):
    groups = {}

    for row in all_results:
        if row['split'] != split:
            continue

        if load_level is not None and row['load_level'] != load_level:
            continue

        key = (row['add_threshold'], row['remove_threshold'], row['scale_down_patience'])
        groups.setdefault(key, []).append(row)

    summary = {}

    for key, rows in groups.items():
        arrived = sum(row['arrived'] for row in rows)
        on_time = sum(row['on_time'] for row in rows)
        worker_seconds = sum(row['worker_seconds'] for row in rows)
        scaling_changes = sum(row['scaling_changes'] for row in rows)

        duration = sum(300 for row in rows)

        summary[key] = {'on_time_percentage': 100 * on_time / arrived if arrived else 0,
            'average_workers': worker_seconds / duration,
            'scaling_changes': scaling_changes / len(rows)}

    return summary


# Figure 1: On-time percentage for every threshold pair.
def plot_threshold_heatmap(all_results, output_folder):
    summary = summarize_results(all_results)

    adds = sorted(set(key[0] for key in summary))
    removes = sorted(set(key[1] for key in summary))

    matrix = np.full((len(adds), len(removes)), np.nan)

    for (add, remove), values in summary.items():
        i = adds.index(add)
        j = removes.index(remove)
        matrix[i, j] = values['on_time_percentage']

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap='viridis',
        vmin=0,
        vmax=100,
        aspect='auto'
    )

    for i in range(len(adds)):
        for j in range(len(removes)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.1f}%',ha='center', va='center', color='white')

    ax.set_xticks(range(len(removes)), removes)
    ax.set_yticks(range(len(adds)), adds)
    ax.set_xlabel('Remove threshold (frames)')
    ax.set_ylabel('Add threshold (frames)')
    ax.set_title('Validation: On-Time Performance')

    fig.colorbar(image, ax=ax, label='On-time frames (%)')
    fig.tight_layout()
    fig.savefig(output_folder / 'threshold_heatmap.png', dpi=200)
    plt.close(fig)


# Figure 2: Deadline performance versus worker usage.
def plot_performance_vs_resources(all_results, output_folder):
    summary = summarize_results(all_results)

    fig, ax = plt.subplots(figsize=(8, 5))

    for (add, remove), values in summary.items():
        x = values['average_workers']
        y = values['on_time_percentage']

        ax.scatter(x, y)
        ax.annotate(f'{add}/{remove}', (x, y),xytext=(5, 5), textcoords='offset points',fontsize=9)

    ax.set_xlabel('Average allocated workers')
    ax.set_ylabel('On-time frames (%)')
    ax.set_title('Validation: Performance vs Resource Usage')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / 'performance_vs_resources.png', dpi=200)
    plt.close(fig)


# Figure 3: Compare standard and stress workload performance.
def plot_standard_vs_stress(all_results, output_folder):
    standard = summarize_results(all_results, load_level='standard')
    stress = summarize_results(all_results, load_level='stress')

    keys = sorted(set(standard) & set(stress))
    labels = [f'{add}/{remove}' for add, remove in keys]

    x = np.arange(len(keys))
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - width / 2,[standard[k]['on_time_percentage'] for k in keys],width, label='Standard')

    ax.bar(x + width / 2,[stress[k]['on_time_percentage'] for k in keys],width, label='Stress')

    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel('Add / Remove thresholds')
    ax.set_ylabel('On-time frames (%)')
    ax.set_title('Validation: Standard vs Stress Workloads')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_folder / 'standard_vs_stress.png', dpi=200)
    plt.close(fig)



def analyze_scaling_actions(all_results):
    print(' Validation Scaling Analysis')

    groups = {}
    for row in all_results:
        if row['split'] != 'validation':
            continue

        key = (row['add_threshold'], row['remove_threshold'])
        groups.setdefault(key, []).append(row)

    print('Add/Remove | Avg Adds | Avg Removes | Avg Blocked')

    for (add, remove), rows in sorted(groups.items()):
        avg_adds = sum(r['add_actions'] for r in rows) / len(rows)
        avg_removes = sum(r['remove_actions'] for r in rows) / len(rows)
        avg_blocked = sum(r['blocked_actions'] for r in rows) / len(rows)

        print(f'{add}/{remove}',f'{avg_adds:1f}',f'{avg_removes:1f}', f'{avg_blocked:.1f}')



if all_results:
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save grid-search results
    with open(output_folder / 'comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print('Finished', len(all_results), 'simulations')
    print('Results saved in:', output_folder)

    # Generate the three analysis figures
    figures_folder = output_folder / 'figures'
    figures_folder.mkdir(parents=True, exist_ok=True)

    plot_threshold_heatmap(all_results, figures_folder)
    plot_performance_vs_resources(all_results, figures_folder)
    plot_standard_vs_stress(all_results, figures_folder)
    analyze_scaling_actions(all_results)
    print('Analysis figures saved in:', figures_folder)

else:
    print('No workload files found.')


