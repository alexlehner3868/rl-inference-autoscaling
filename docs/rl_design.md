# RL Design

## State

The DQN will use five inputs:

- queue length
- oldest frame waiting time
- recent frame arrival rate
- number of active workers
- number of workers currently starting

These describe the current demand, backlog, and available processing capacity. The exact ranges and any normalization will be decided once the simulator and YOLO profiling are finished.

## Actions

There are three actions:

- 0: remove one worker
- 1: keep the current number of workers
- 2: add one worker

The simulator will enforce the minimum and maximum worker limits.

## Reward

The reward will balance latency performance against resource use. The initial form is:

    reward =
        on_time_reward * on_time_frames
        - late_penalty * late_frames
        - worker_cost * worker_seconds
        - scaling_penalty * scaled

The reward weights are preliminary and will be tuned later using the validation workloads.

## Metrics

The main metrics are:

- SLA success rate
- worker-seconds

We will also track:

- total frames
- on-time frames
- late frames
- dropped frames
- mean latency
- p95 latency
- p99 latency
- scale-up actions
- scale-down actions
- average active workers
- peak active workers
- per-camera SLA results

All controllers will use the same metrics.

## Simulator Interface

The RL environment currently expects the simulator to expose:

    queue_length
    oldest_wait_time
    recent_arrival_rate
    active_workers
    starting_workers

The simulator should have:

    reset(seed=None)
    step(action)

For each step, the environment needs:

    on_time_frames
    late_frames
    worker_seconds
    scaled
    terminated
    truncated

This interface can be adjusted once the simulator implementation is finalized.

## DQN

The initial implementation uses Stable-Baselines3 DQN with an MLP policy. DQN hyperparameters will be tuned later using the validation workloads after the simulator has been validated. The held-out test workloads will not be used for tuning.