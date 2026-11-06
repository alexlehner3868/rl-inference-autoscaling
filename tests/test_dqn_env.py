import numpy as np
import pytest

from agent.dqn_env import InferenceScalingEnv


CALIBRATION = {
    "service_time_mean_seconds": 0.05,
    "startup_mean_seconds": 0.2,
}


WORKLOAD = {
    "name": "test_workload",
    "camera_fps": 10.0,
    "decision_seconds": 0.1,
    "duration_seconds": 0.5,
    "intervals": [
        {
            "start_seconds": 0.0,
            "end_seconds": 0.2,
            "camera_ids": [0],
        },
        {
            "start_seconds": 0.2,
            "end_seconds": 0.5,
            "camera_ids": [0, 1],
        },
    ],
}


def make_env():
    return InferenceScalingEnv(
        workloads=[WORKLOAD],
        calibration=CALIBRATION,
    )


def test_reset_returns_real_simulator_observation():
    env = make_env()

    observation, info = env.reset(seed=1)

    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(
        observation
    )

    assert observation[2] == pytest.approx(
        10.0
    )

    assert observation[3] == pytest.approx(
        1.0
    )

    assert info["workload"] == "test_workload"


def test_numeric_add_action_maps_to_simulator_add():
    env = make_env()

    env.reset(seed=1)

    _, _, _, _, info = env.step(
        env.ADD_WORKER
    )

    assert info["action"] == "add"
    assert info["action_applied"] is True
    assert info["scaled"] is True
    assert info["workers"] == 2


def test_hold_action_does_not_count_as_scaling():
    env = make_env()

    env.reset(seed=1)

    _, reward, _, _, info = env.step(
        env.KEEP_CAPACITY
    )

    assert info["action"] == "hold"
    assert info["scaled"] is False
    assert isinstance(reward, float)


def test_observation_updates_when_workload_demand_changes():
    env = make_env()

    env.reset(seed=1)

    env.step(
        env.KEEP_CAPACITY
    )

    observation, _, _, _, _ = env.step(
        env.KEEP_CAPACITY
    )

    assert observation[2] == pytest.approx(
        20.0
    )


def test_episode_ends_at_workload_duration():
    env = make_env()

    env.reset(seed=1)

    terminated = False

    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(
            env.KEEP_CAPACITY
        )

    assert terminated is True
    assert truncated is False


def test_invalid_action_raises_error():
    env = make_env()

    env.reset(seed=1)

    with pytest.raises(ValueError):
        env.step(3)