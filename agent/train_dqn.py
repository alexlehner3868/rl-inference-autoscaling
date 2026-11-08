from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from agent.dqn_env import InferenceScalingEnv


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    ROOT / "results" / "models" / "dqn_autoscaler"
)


def load_workloads(split: str) -> list[dict]:
    folder = ROOT / "workloads" / split
    files = sorted(folder.glob("*.json"))

    if not files:
        raise ValueError(f"No workloads found in {folder}")

    return [
        json.loads(path.read_text())
        for path in files
    ]


def load_calibration() -> dict:
    path = (
        ROOT
        / "results"
        / "calibration"
        / "simulator_parameters.json"
    )

    return json.loads(path.read_text())


def make_env(
    split: str = "train",
) -> InferenceScalingEnv:
    return InferenceScalingEnv(
        workloads=load_workloads(split),
        calibration=load_calibration(),
    )


def create_dqn(
    env: InferenceScalingEnv,
    seed: int = 0,
    *,
    algorithm: type[DQN] = DQN,
) -> DQN:
    return algorithm(
        "MlpPolicy", env,
        learning_starts=1000,
        buffer_size=50000,
        seed=seed,
        verbose=1,
)


def train_dqn(
    total_timesteps: int = 30000,
    seed: int = 0,
    save_path: str | Path = DEFAULT_MODEL_PATH,
) -> DQN:
    env = make_env("train")

    check_env(
        env,
        warn=True,
    )

    model = create_dqn(
        env,
        seed=seed,
    )

    model.learn(
        total_timesteps=total_timesteps,
    )

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        str(save_path),
    )

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    args = parser.parse_args()

    train_dqn(
        total_timesteps=args.timesteps,
        seed=args.seed,
        save_path=args.output,
    )

    print(
        f"Saved DQN model to {args.output}.zip"
    )


if __name__ == "__main__":
    main()
