from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import DQN

from agent.dqn_env import InferenceScalingEnv


DEFAULT_MODEL_PATH = Path("results/models/dqn_autoscaler")


def create_dqn(env: InferenceScalingEnv) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        verbose=1,
    )


def train_dqn(
    simulator: Any,
    total_timesteps: int,
    save_path: str | Path = DEFAULT_MODEL_PATH,
) -> DQN:
    env = InferenceScalingEnv(simulator)

    model = create_dqn(env)
    model.learn(total_timesteps=total_timesteps)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(save_path))

    return model