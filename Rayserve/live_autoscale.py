

import argparse
import csv
from email import parser
from html import parser
import json
import math
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from Simulator.Simulator_queue import SimulatorConfig, calculate_reward

# Settings 
add_threshold = 3
remove_threshold = 1
scale_down_patience = 10
max_workers = SimulatorConfig.max_workers
queue_size = 50
deadline = 1.0

workload_file = root / 'workloads/validation/flash_crowds_01.json'
video_file = root / 'TrafficCamData/Dataset_A/cam_1.mp4'
dashboard = 'http://127.0.0.1:8265'
inference_url = 'http://127.0.0.1:8000/'


def load_frames(video, count=24):
    video = cv2.VideoCapture(str(video))
    frames = []
    for _ in range(count):
        success, frame = video.read()
        if not success:
            break
        success, image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            frames.append(image.tobytes())

    video.release()
    if not frames:
        raise ValueError('Could not read video frames. Check Git LFS.')
    return frames


# Make one arrival event for every frame in the workload.
def make_events(workload, duration):
    events = []
    period = 1 / workload['camera_fps']

    for interval in workload['intervals']:
        start = interval['start_seconds']
        end = min(interval['end_seconds'], duration)
        frame_number = math.ceil(start / period - 1e-9)  

        while frame_number * period < end - 1e-9:
            seconds = round(frame_number * period, 6)
            for camera_id in interval['camera_ids']:
                events.append((seconds, camera_id))
            frame_number += 1
    events.sort()
    return events


def get_demand(workload, seconds):
    for interval in workload['intervals']:
        if interval['start_seconds'] <= seconds < interval['end_seconds']:
            return len(interval['camera_ids']) * workload['camera_fps']
    return 0


# Ask Ray Serve how many replicas are running or starting.
def get_workers(address=dashboard, require_external_scaler=True):
    response = requests.get(address + '/api/serve/applications/', timeout=5)
    response.raise_for_status()
    applications = response.json()['applications']
    app = applications.get('camera-app')

    if app is None or app['status'] == 'DEPLOY_FAILED':
        raise RuntimeError('camera-app is not running. Check serve status.')
    if bool(app.get('external_scaler_enabled')) != require_external_scaler:
        config_name = 'serve_config.yaml' if require_external_scaler else 'serve_autoscaling_config.yaml'
        raise RuntimeError(f'Deploy Rayserve/{config_name} for this controller')

    deployment = app['deployments']['YOLOWorker']
    autoscaling_config = deployment.get('deployment_config', {}).get('autoscaling_config')
    if not require_external_scaler and not autoscaling_config:
        raise RuntimeError('YOLOWorker has no Ray Serve autoscaling_config')
    replicas = deployment['replicas']
    ready = sum(worker['state'] == 'RUNNING' for worker in replicas)
    target = int(deployment['target_num_replicas'])

    return {'ready': ready, 'starting': max(0, target - ready), 'allocated': len(replicas),
            'target': target, 'autoscaling_config': autoscaling_config}


# Request a new number of YOLO replicas.
def set_workers(workers, address=dashboard):
    url = address + '/api/v1/applications/camera-app/deployments/YOLOWorker/scale'
    response = requests.post(url, json={'target_num_replicas': workers}, timeout=10)
    response.raise_for_status()


def wait_for_workers(workers, address=dashboard, timeout=120, require_external_scaler=True):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        status = get_workers(address, require_external_scaler=require_external_scaler)
        if status['target'] == workers and status['ready'] == workers and status['allocated'] == workers:
            return status
        time.sleep(0.5)
    raise TimeoutError('Could not start requested Ray Serve replicas. Check serve status.')


# Each request runs in a thread, so several YOLO replicas can work simultaneously.
def send_frame(jpeg, url=inference_url, timeout=4):
    try:
        response = requests.post(url, data=jpeg, headers={'Content-Type': 'image/jpeg'}, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if 'model_ms' not in result:
            raise ValueError('YOLO response does not contain model_ms')

        return {'success': True, 'finish': time.monotonic(), 'model_ms': result['model_ms'],
                'replica_pid': result.get('replica_pid'), 'error': ''}
    except Exception as error:
        return {'success': False, 'finish': time.monotonic(), 'model_ms': None,
                'replica_pid': None, 'error': str(error)}


# Save the experiment in the same format as the earlier script.
def save_results(folder, summary, records, history, scaling):
    folder.mkdir(parents=True, exist_ok=False)

    with open(folder / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    for name, rows in [('frames.csv', records), ('history.csv', history), ('scaling.csv', scaling)]:
        if not rows:
            continue
        with open(folder / name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    if history:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        times = [row['seconds'] for row in history]
        axes[0].step(times, [row['demand_fps'] for row in history], where='post', label='Demand (FPS)')
        axes[1].step(times, [row['queue_length'] for row in history], where='post', label='Queue length')
        axes[1].step(times, [row['ready_replicas'] for row in history], where='post', label='Ready replicas')
        axes[1].step(times, [row['target_replicas'] for row in history], where='post', label='Target replicas')
        axes[0].set_ylabel('Frames per second')
        axes[1].set_ylabel('Frames / replicas')
        axes[1].set_xlabel('Time (seconds)')
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(folder / 'live_timeline.png', dpi=180)
        plt.close(fig)


# Run the live workload with the selected controller.
def run_experiment(args, workload, images, model=None):
    normalize_observation = None
    if args.controller == 'ppo' and args.normalize_ppo_observation:
        from agent.ppo_observation import normalize_ppo_observation
        normalize_observation = normalize_ppo_observation

    duration = workload['duration_seconds'] if args.duration == 0 else min(args.duration, workload['duration_seconds'])
    events = make_events(workload, duration)
    if not events:
        raise ValueError('No frames in this time range. Try a longer duration.')

    # Start each experiment from a known number of replicas.
    initial_workers = args.workers if args.controller == 'fixed' else 1
    if args.controller == 'ray_auto':
        status = get_workers(args.dashboard, require_external_scaler=False)
        configured_min = status['autoscaling_config'].get('min_replicas')
        configured_max = status['autoscaling_config'].get('max_replicas')
        if configured_min != 1 or configured_max != args.max_workers:
            raise RuntimeError(f'Ray Serve autoscaling range is {configured_min}–{configured_max}; 'f'expected 1–{args.max_workers}')
        status = wait_for_workers(1, args.dashboard, timeout=args.ready_timeout, require_external_scaler=False)
    else:
        get_workers(args.dashboard)
        set_workers(initial_workers, args.dashboard)
        status = wait_for_workers(initial_workers, args.dashboard, timeout=args.ready_timeout)
    ray_autoscaling_config = status['autoscaling_config'] if args.controller == 'ray_auto' else None
    target = initial_workers
    low_queue_steps = 0
    predictive_demand = deque(maxlen=args.forecast_window)
    predictive_previous_arrived = 0
    predictive_low_steps = 0

    queue = deque()
    pending = {}
    records = []
    history = []
    scaling = []
    service_times = []
    counts = {'arrived': 0, 'on_time': 0, 'late': 0, 'dropped_full': 0, 'dropped_deadline': 0, 'failed': 0}
    next_event = 0
    frame_id = 0
    next_decision = 0
    next_history = 0
    next_status = 0
    worker_seconds = 0
    ready_worker_seconds = 0
    last_sample = 0
    last_allocated = status['allocated']
    last_ready = status['ready']
    max_queue_length = 0
    blocked = 0

    # The built-in autoscaler must see requests waiting at the Serve proxy.
    pool_workers = args.queue_size + args.max_workers if args.controller == 'ray_auto' else args.max_workers
    pool = ThreadPoolExecutor(max_workers=pool_workers)
    start = time.monotonic()
    print('Controller:', args.controller, 'Duration:', duration, 'Frames:', len(events), flush=True)

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= duration + args.deadline:
                break

            # Check how many actual YOLO replicas are available.
            if elapsed >= next_status:
                measured_until = min(elapsed, duration)
                sample_seconds = measured_until - last_sample
                worker_seconds += last_allocated * sample_seconds
                ready_worker_seconds += last_ready * sample_seconds
                last_sample = measured_until
                status = get_workers(args.dashboard, require_external_scaler=args.controller != 'ray_auto')
                last_allocated = status['allocated']
                last_ready = status['ready']
                next_status += 0.5
                if args.controller == 'ray_auto':
                    if elapsed < duration and status['target'] != target:
                        scaling.append({'seconds': round(elapsed, 2), 'action': 'ray_auto',
                            'applied': 1, 'target_before': target, 'target_after': status['target'],
                            'ready_replicas': status['ready'], 'queue_length': len(queue)})
                    target = status['target']

            # Record results from requests that have finished.
            for task, frame in list(pending.items()):
                if not task.done():
                    continue

                result = task.result()
                latency = result['finish'] - frame['arrival']
                if not result['success']:
                    outcome = 'failed'
                elif latency <= args.deadline:
                    outcome = 'on_time'
                else:
                    outcome = 'late'

                counts[outcome] += 1
                records.append({'frame_id': frame['id'], 'camera_id': frame['camera_id'],
                    'status': outcome, 'latency_ms': round(latency * 1000, 2),
                    'model_ms': result['model_ms'], 'replica_pid': result['replica_pid'],
                    'error': result['error']})
                if result['model_ms'] is not None:
                    service_times.append(result['model_ms'] / 1000)
                del pending[task]

            # Drop waiting frames once they miss the deadline.
            while queue and now - queue[0]['arrival'] >= args.deadline:
                frame = queue.popleft()
                counts['dropped_deadline'] += 1
                records.append({'frame_id': frame['id'], 'camera_id': frame['camera_id'],
                    'status': 'dropped_deadline', 'latency_ms': None,
                    'model_ms': None, 'replica_pid': None, 'error': ''})

            # Add new frames according to the workload arrival schedule.
            while next_event < len(events) and events[next_event][0] <= elapsed:
                scheduled, camera_id = events[next_event]
                frame = {'id': frame_id, 'camera_id': camera_id, 'arrival': start + scheduled,'jpeg': images[frame_id % len(images)]}
                frame_id += 1
                next_event += 1
                counts['arrived'] += 1

                # If the computer fell behind, the frame may already be too late.
                if now - frame['arrival'] >= args.deadline:
                    counts['dropped_deadline'] += 1
                    records.append({'frame_id': frame['id'], 'camera_id': camera_id,
                        'status': 'dropped_deadline', 'latency_ms': None,
                        'model_ms': None, 'replica_pid': None, 'error': ''})
                elif (len(queue) + len(pending) >= args.queue_size + status['ready']
                      if args.controller == 'ray_auto' else len(queue) >= args.queue_size):
                    counts['dropped_full'] += 1
                    records.append({'frame_id': frame['id'], 'camera_id': camera_id,
                        'status': 'dropped_full', 'latency_ms': None,
                        'model_ms': None, 'replica_pid': None, 'error': ''})
                else:
                    queue.append(frame)

            max_queue_length = max(max_queue_length, len(queue))
            desired_target = None
            # Every decision interval, decide whether to add, remove, or hold.
            if args.controller != 'ray_auto' and elapsed < duration and elapsed >= next_decision:
                waiting = len(queue)
                oldest = now - queue[0]['arrival'] if queue else 0
                demand_fps = get_demand(workload, elapsed)
                action = 'hold'

                if args.controller == 'queue':
                    if waiting >= args.add_threshold and target < args.max_workers:
                        action = 'add'
                        low_queue_steps = 0
                    elif waiting <= args.remove_threshold and target > 1:
                        all_ready = status['ready'] == target and status['starting'] == 0
                        service_seconds = sum(service_times[-100:]) / len(service_times[-100:]) if service_times else args.calibrated_service
                        capacity_after_remove = (target - 1) / service_seconds

                        if all_ready and demand_fps <= capacity_after_remove:
                            low_queue_steps += 1
                            if low_queue_steps >= args.patience:
                                action = 'remove'
                                low_queue_steps = 0
                        else:
                            low_queue_steps = 0
                    else:
                        low_queue_steps = 0

                elif args.controller == 'predictive':
                    # The first decision is at time zero, before a complete
                    # decision interval of arrivals has been observed.
                    if next_decision > 0:
                        observed_fps = (counts['arrived'] - predictive_previous_arrived) / workload['decision_seconds']
                        predictive_demand.append(observed_fps)
                    predictive_previous_arrived = counts['arrived']

                    forecast_fps = (predictive_demand[-1] if predictive_demand else 0.0)
                    if len(predictive_demand) >= 2:
                        times = (np.arange(len(predictive_demand))* workload['decision_seconds'])
                        slope, intercept = np.polyfit(times, list(predictive_demand), 1)
                        forecast_fps = max(0.0,float(intercept + slope * (times[-1] + args.calibrated_startup)),)

                    forecast_target = max(1,min(args.max_workers,math.ceil(forecast_fps * args.calibrated_service - 1e-9)))

                    if forecast_target > target and target < args.max_workers:
                        action = 'add'
                        predictive_low_steps = 0
                    elif (forecast_target < target and waiting == 0 and status['starting'] == 0):
                        predictive_low_steps += 1
                        if (predictive_low_steps >= args.predictive_down_steps and status['ready'] > 1):
                            action = 'remove'
                            predictive_low_steps = 0
                    else:
                        predictive_low_steps = 0

                elif args.controller in ('dqn', 'ppo'):
                    observation = np.array([waiting, oldest, demand_fps, status['ready'], status['starting']], dtype=np.float32)
                    if normalize_observation is not None:
                        observation = normalize_observation(observation)
                    selected, _ = model.predict(observation, deterministic=True)


                    if args.controller == 'ppo' and args.ppo_target_actions:
                        desired_target = int(selected) + 1
                        action = 'set_target'
                    else:
                        action = ['remove', 'hold', 'add'][int(selected)]

                previous = target
                if desired_target is not None:
                    if desired_target != target:
                        target = desired_target
                        set_workers(target, args.dashboard)

                elif action == 'add':
                    if target < args.max_workers:
                        target += 1
                        set_workers(target, args.dashboard)
                    else:
                        blocked += 1
                elif action == 'remove':
                    if target > 1 and status['ready'] > 1:
                        target -= 1
                        set_workers(target, args.dashboard)
                    else:
                        blocked += 1

                scaling.append({'seconds': round(elapsed, 2), 'action': action,
                    'applied': int(target != previous), 'target_before': previous,
                    'target_after': target, 'ready_replicas': status['ready'], 'queue_length': waiting})
                next_decision += workload['decision_seconds']

            # External controllers send one request per ready replica. The
            # built-in autoscaler also receives requests waiting at Serve.
            dispatch_limit = args.queue_size + status['ready'] if args.controller == 'ray_auto' else status['ready']
            while queue and len(pending) < dispatch_limit:
                frame = queue.popleft()
                if time.monotonic() - frame['arrival'] >= args.deadline:
                    counts['dropped_deadline'] += 1
                    records.append({'frame_id': frame['id'], 'camera_id': frame['camera_id'],
                        'status': 'dropped_deadline', 'latency_ms': None,
                        'model_ms': None, 'replica_pid': None, 'error': ''})
                else:
                    task = pool.submit(send_frame, frame['jpeg'], args.inference_url, args.request_timeout)
                    pending[task] = frame

            if elapsed <= duration and elapsed >= next_history:
                history.append({'seconds': round(elapsed, 2), 'demand_fps': get_demand(workload, elapsed),
                    'queue_length': len(queue), 'inflight': len(pending),
                    'target_replicas': target, 'ready_replicas': status['ready'],
                    'starting_replicas': status['starting'], 'allocated_replicas': status['allocated'],
                    'arrived': counts['arrived'], 'on_time': counts['on_time'],
                    'late': counts['late'], 'failed': counts['failed'],
                    'dropped_full': counts['dropped_full'], 'dropped_deadline': counts['dropped_deadline']})
                if args.verbose:
                    print('Time:', round(elapsed, 1), 'Demand:', get_demand(workload, elapsed),
                        'Queue:', len(queue), 'Ready:', status['ready'], 'Target:', target, flush=True)
                next_history += 1

            time.sleep(0.01)
    finally:
        # Wait for outstanding HTTP calls to terminate before closing the thread pool.
        pool.shutdown(wait=True)

    remaining_seconds = max(0, duration - last_sample)
    worker_seconds += last_allocated * remaining_seconds
    ready_worker_seconds += last_ready * remaining_seconds
    unfinished = len(queue) + len(pending)
    completed = sum(counts[key] for key in ['on_time', 'late', 'dropped_full', 'dropped_deadline', 'failed'])
    if counts['arrived'] != completed + unfinished:
        raise RuntimeError('Frame accounting mismatch')

    latencies = [row['latency_ms'] / 1000 for row in records if row['status'] in ('on_time', 'late')]
    summary = {'controller': args.controller, 'workload': args.workload.stem,
        'duration_seconds': duration, 'scheduled_frames': len(events), **counts,
        'on_time_percentage': 100 * counts['on_time'] / counts['arrived'],
        'average_workers': worker_seconds / duration, 'worker_seconds': worker_seconds,
        'average_ready_workers': ready_worker_seconds / duration,
        'ready_worker_seconds': ready_worker_seconds,
        'unfinished_frames': unfinished,
        'p95_latency_seconds': float(np.percentile(latencies, 95)) if latencies else None,
        'max_queue_length': max_queue_length,
        'scaling_changes': sum(abs(row['target_after'] - row['target_before']) for row in scaling),
        'blocked_scaling_actions': blocked,
        'add_actions': sum(max(0, row['target_after'] - row['target_before']) for row in scaling),
        'remove_actions': sum(max(0, row['target_before'] - row['target_after']) for row in scaling),
        'frame_deadline_seconds': args.deadline, 'request_timeout_seconds': args.request_timeout,
        'ray_autoscaling_config': ray_autoscaling_config,
        'http_concurrency_limit': pool_workers,
        'queue_capacity_frames': args.queue_size,
        'max_workers': args.max_workers, 'initial_replicas': initial_workers,
        'video': str(args.video), 'cached_video_frames': len(images)}
    summary['total_reward'] = calculate_reward(
        counts['on_time'], counts['late'] + counts['dropped_full'] + counts['dropped_deadline'] + counts['failed'],
        worker_seconds, summary['scaling_changes'])
    return summary, records, history, scaling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=Path, default=workload_file)
    parser.add_argument('--video', type=Path, default=video_file)
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--max-workers', type=int, default=max_workers)
    parser.add_argument('--add-threshold', type=int, default=add_threshold)
    parser.add_argument('--remove-threshold', type=int, default=remove_threshold)
    parser.add_argument('--patience', type=int, default=scale_down_patience)
    parser.add_argument('--queue-size', type=int, default=queue_size)
    parser.add_argument('--deadline', type=float, default=deadline)
    parser.add_argument('--calibrated-service', type=float, default=0.07197132544888889)
    parser.add_argument('--calibrated-startup', type=float, default=2.3057613082)
    parser.add_argument('--forecast-window', type=int, default=10)
    parser.add_argument('--predictive-down-steps', type=int, default=5)
    parser.add_argument('--model', type=Path, default=None)
    parser.add_argument('--dashboard', default=dashboard)
    parser.add_argument('--inference-url', default=inference_url)
    parser.add_argument('--request-timeout', type=float, default=4)
    parser.add_argument('--ready-timeout', type=float, default=120)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--normalize-ppo-observation', action='store_true')
    parser.add_argument('--ppo-target-actions', action='store_true')
    parser.add_argument('--controller',choices=['fixed', 'queue', 'predictive', 'ray_auto', 'dqn', 'ppo'],default='queue')
    args = parser.parse_args()
    if (args.calibrated_service <= 0 or args.calibrated_startup <= 0 or args.forecast_window < 1 or args.predictive_down_steps < 1):
        parser.error('Invalid predictive settings')
    if args.ppo_target_actions:
        if args.controller != 'ppo':
            parser.error('--ppo-target-actions requires --controller ppo')
        if args.max_workers != 10:
            parser.error('Target-action PPO requires --max-workers 10')
        args.normalize_ppo_observation = True

    if args.max_workers < 1 or not 1 <= args.workers <= args.max_workers:
        parser.error('Invalid worker limits')
    if args.queue_size < 1 or args.deadline <= 0 or args.duration < 0 or args.patience < 1 or args.request_timeout <= 0:
        parser.error('Invalid queue, deadline, duration, patience, or request timeout')
    if args.request_timeout < args.deadline:
        parser.error('--request-timeout must be at least --deadline so an on-time response is not cut off')
    if args.ready_timeout <= 0:
        parser.error('--ready-timeout must be positive')
    if args.add_threshold < 1 or not 0 <= args.remove_threshold < args.add_threshold:
        parser.error('Require 0 <= remove threshold < add threshold')
    if args.controller in ('dqn', 'ppo') and args.model is None:
        parser.error('DQN and PPO require --model checkpoint.zip')

    with open(args.workload) as f:
        workload = json.load(f)
    images = load_frames(args.video)
    model = None

    if args.controller == 'dqn':
        from stable_baselines3 import DQN
        model = DQN.load(args.model)
    elif args.controller == 'ppo':
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

    try:
        summary, records, history, scaling = run_experiment(args, workload, images, model)
    finally:
        # External controllers return to one worker; Ray's autoscaler controls itself.
        if args.controller != 'ray_auto':
            set_workers(1, args.dashboard)

    folder = args.output or (root / 'results/rayserve_live' / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / (args.controller + '_' + args.workload.stem))
    save_results(folder, summary, records, history, scaling)

    print('\nValidation Summary')
    print('On-time:', round(summary['on_time_percentage'], 2), '%')
    print('On-time frames:', summary['on_time'], '/', summary['arrived'])
    print('Completed late:', summary['late'])
    print('Failed requests:', summary['failed'])
    print('Unfinished at cutoff:', summary['unfinished_frames'])
    print('Average workers:', round(summary['average_workers'], 2))
    print('Total reward:', round(summary['total_reward'], 2))
    print('Average ready workers:', round(summary['average_ready_workers'], 2))
    print('Dropped (full/deadline):', summary['dropped_full'], '/', summary['dropped_deadline'])
    print('Scaling changes:', summary['scaling_changes'])
    print('Results saved in:', folder)


if __name__ == '__main__':
    main()
