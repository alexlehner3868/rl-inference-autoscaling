import numpy as np
import pytest

from agent.dqn_env import InferenceScalingEnv


class MockSimulator:
    def __init__(self):
        self.reset()

    def reset(self, seed=None):
        self.queue_length = 5
        self.oldest_wait_time = 0.2
        self.recent_arrival_rate = 10.0
        self.active_workers = 2
        self.starting_workers = 0

    def step(self, action):
        old_workers = self.active_workers

        if action == 0 and self.active_workers > 1:
            self.active_workers -= 1
        elif action == 2:
            self.active_workers += 1

        scaled = self.active_workers != old_workers

        self.queue_length = max(
            0,
            self.queue_length - self.active_workers,
        )

        return {
            "on_time_frames": 2,
            "late_frames": 0,
            "worker_seconds": float(self.active_workers),
            "scaled": scaled,
            "terminated": False,
            "truncated": False,
        }


def test_reset():
    env = InferenceScalingEnv(MockSimulator())

    observation, info = env.reset(seed=42)

    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info == {}


def test_keep_capacity():
    env = InferenceScalingEnv(MockSimulator())
    env.reset()

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation.shape == (5,)
    assert isinstance(reward, float)
    assert info["scaled"] is False
    assert terminated is False
    assert truncated is False


def test_add_worker():
    env = InferenceScalingEnv(MockSimulator())
    env.reset()

    _, _, _, _, info = env.step(2)

    assert env.simulator.active_workers == 3
    assert info["scaled"] is True


def test_remove_worker():
    env = InferenceScalingEnv(MockSimulator())
    env.reset()

    _, _, _, _, info = env.step(0)

    assert env.simulator.active_workers == 1
    assert info["scaled"] is True


def test_late_frames_reduce_reward():
    env = InferenceScalingEnv(MockSimulator())

    good = env._calculate_reward(
        {
            "on_time_frames": 2,
            "late_frames": 0,
            "worker_seconds": 2.0,
            "scaled": False,
        }
    )

    late = env._calculate_reward(
        {
            "on_time_frames": 1,
            "late_frames": 1,
            "worker_seconds": 2.0,
            "scaled": False,
        }
    )

    assert late < good


def test_invalid_action():
    env = InferenceScalingEnv(MockSimulator())

    with pytest.raises(ValueError):
        env.step(3)