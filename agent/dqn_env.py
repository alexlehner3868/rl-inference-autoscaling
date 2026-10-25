from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class InferenceScalingEnv(gym.Env):
    metadata = {"render_modes": []}

    REMOVE_WORKER = 0
    KEEP_CAPACITY = 1
    ADD_WORKER = 2

    def __init__(
        self,
        simulator: Any,
        on_time_reward: float = 1.0,
        late_penalty: float = 1.0,
        worker_cost: float = 1.0,
        scaling_penalty: float = 1.0,
    ):
        super().__init__()

        self.simulator = simulator

        self.on_time_reward = on_time_reward
        self.late_penalty = late_penalty
        self.worker_cost = worker_cost
        self.scaling_penalty = scaling_penalty

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.full(5, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        self.simulator.reset(seed=seed)

        return self._get_observation(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        results = self.simulator.step(action)

        observation = self._get_observation()
        reward = self._calculate_reward(results)

        terminated = bool(results.get("terminated", False))
        truncated = bool(results.get("truncated", False))

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            dict(results),
        )

    def _get_observation(self) -> np.ndarray:
        return np.array(
            [
                self.simulator.queue_length,
                self.simulator.oldest_wait_time,
                self.simulator.recent_arrival_rate,
                self.simulator.active_workers,
                self.simulator.starting_workers,
            ],
            dtype=np.float32,
        )

    def _calculate_reward(self, results: dict) -> float:
        on_time_frames = float(results.get("on_time_frames", 0))
        late_frames = float(results.get("late_frames", 0))
        worker_seconds = float(results.get("worker_seconds", 0))
        scaled = bool(results.get("scaled", False))

        return (
            self.on_time_reward * on_time_frames
            - self.late_penalty * late_frames
            - self.worker_cost * worker_seconds
            - self.scaling_penalty * float(scaled)
        )