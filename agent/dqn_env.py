from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Simulator.Simulator_queue import Simulator, SimulatorConfig


class InferenceScalingEnv(gym.Env):
    metadata = {"render_modes": []}

    REMOVE_WORKER = 0
    KEEP_CAPACITY = 1
    ADD_WORKER = 2

    ACTIONS = {
        REMOVE_WORKER: "remove",
        KEEP_CAPACITY: "hold",
        ADD_WORKER: "add",
    }

    def __init__(
        self,
        workloads: list[dict[str, Any]],
        calibration: dict[str, Any],
        initial_workers: int = 1,
        max_workers: int = 4,
        deadline_seconds: float = 1.0,
        queue_capacity_frames: int = 50,
        on_time_reward: float = 1.0,
        failed_frame_penalty: float = 2.0,
        worker_cost: float = 0.1,
        scaling_penalty: float = 0.25,
    ):
        super().__init__()

        if not workloads:
            raise ValueError("At least one workload is required")

        self.workloads = workloads
        self.calibration = calibration
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        self.deadline_seconds = deadline_seconds
        self.queue_capacity_frames = queue_capacity_frames

        self.on_time_reward = on_time_reward
        self.failed_frame_penalty = failed_frame_penalty
        self.worker_cost = worker_cost
        self.scaling_penalty = scaling_penalty

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.full(5, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.workload = None
        self.simulator = None
        self.interval_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        if options is not None and "workload_index" in options:
            workload_index = int(options["workload_index"])
        else:
            workload_index = int(
                self.np_random.integers(len(self.workloads))
            )

        if not 0 <= workload_index < len(self.workloads):
            raise ValueError("Invalid workload index")

        self.workload = self.workloads[workload_index]
        self.interval_index = 0

        config = SimulatorConfig(
            camera_fps=self.workload["camera_fps"],
            service_seconds=self.calibration[
                "service_time_mean_seconds"
            ],
            startup_seconds=self.calibration[
                "startup_mean_seconds"
            ],
            deadline_seconds=self.deadline_seconds,
            decision_seconds=self.workload["decision_seconds"],
            initial_workers=self.initial_workers,
            min_serving_workers=1,
            max_workers=self.max_workers,
            queue_capacity_frames=self.queue_capacity_frames,
        )

        self.simulator = Simulator(config)

        return self._get_observation(), {
            "workload": self.workload.get("name", "unknown"),
        }

    def step(self, action: int):
        if self.simulator is None or self.workload is None:
            raise RuntimeError("Call reset() before step()")

        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        camera_ids = self._current_camera_ids()
        before = self._counters()

        result = self.simulator.step(
            self.ACTIONS[action],
            camera_ids,
        )

        after = self._counters()

        on_time_frames = (
            after["on_time"] - before["on_time"]
        )

        late_frames = (
            after["late"] - before["late"]
        )

        dropped_frames = (
            after["dropped_deadline"]
            - before["dropped_deadline"]
            + after["dropped_full"]
            - before["dropped_full"]
        )

        interval_worker_seconds = (
            after["worker_seconds"]
            - before["worker_seconds"]
        )

        scaled = (
            action != self.KEEP_CAPACITY
            and bool(result["action_applied"])
        )

        reward = (
            self.on_time_reward * on_time_frames
            - self.failed_frame_penalty
            * (late_frames + dropped_frames)
            - self.worker_cost * interval_worker_seconds
            - self.scaling_penalty * float(scaled)
        )

        terminated = (
            result["time_seconds"]
            >= self.workload["duration_seconds"] - 1e-9
        )

        info = dict(result)

        info.update(
            workload=self.workload.get("name", "unknown"),
            on_time_frames=on_time_frames,
            late_frames=late_frames,
            dropped_frames=dropped_frames,
            interval_worker_seconds=interval_worker_seconds,
            scaled=scaled,
        )

        return (
            self._get_observation(),
            float(reward),
            bool(terminated),
            False,
            info,
        )

    def _current_camera_ids(self):
        if self.simulator is None or self.workload is None:
            return ()

        time_seconds = (
            self.simulator.tick
            * self.simulator.config.tick_seconds
        )

        intervals = self.workload["intervals"]

        while (
            self.interval_index < len(intervals)
            and time_seconds
            >= intervals[self.interval_index]["end_seconds"] - 1e-9
        ):
            self.interval_index += 1

        if self.interval_index >= len(intervals):
            return ()

        return tuple(
            intervals[self.interval_index]["camera_ids"]
        )

    def _get_observation(self) -> np.ndarray:
        if self.simulator is None:
            return np.zeros(5, dtype=np.float32)

        camera_ids = self._current_camera_ids()

        if self.simulator.queue:
            oldest_arrival_tick = self.simulator.queue[0][1]

            oldest_wait_time = (
                self.simulator.tick - oldest_arrival_tick
            ) * self.simulator.config.tick_seconds
        else:
            oldest_wait_time = 0.0

        serving_workers = sum(
            worker["ready_at"] <= self.simulator.tick
            and not worker["retiring"]
            for worker in self.simulator.workers
        )

        starting_workers = sum(
            worker["ready_at"] > self.simulator.tick
            and not worker["retiring"]
            for worker in self.simulator.workers
        )

        demand_fps = (
            len(camera_ids)
            * self.simulator.config.camera_fps
        )

        return np.array(
            [
                len(self.simulator.queue),
                oldest_wait_time,
                demand_fps,
                serving_workers,
                starting_workers,
            ],
            dtype=np.float32,
        )

    def _counters(self):
        return {
            "on_time": self.simulator.on_time,
            "late": self.simulator.late,
            "dropped_deadline": self.simulator.dropped_deadline,
            "dropped_full": self.simulator.dropped_full,
            "worker_seconds": self.simulator.worker_seconds,
        }