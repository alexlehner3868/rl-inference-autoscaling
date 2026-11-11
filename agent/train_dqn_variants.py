from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from agent.dueling_dqn import DuelingDQNPolicy
from agent.train_dqn import (
    ROOT,
    create_dqn,
    make_env,
)


VARIANTS = {
    "vanilla": {
        "policy": "MlpPolicy",
        "n_steps": 1,
    },
    "nstep": {
        "policy": "MlpPolicy",
        "n_steps": 3,
    },
    "dueling": {
        "policy": DuelingDQNPolicy,
        "n_steps": 1,
    },
    "dueling_nstep": {
        "policy": DuelingDQNPolicy,
        "n_steps": 3,
    },
}


def train_variant(
    variant: str,
    total_timesteps: int,
    seed: int,
    on_time_reward: float,
    failed_frame_penalty: float,
    worker_cost: float,
    scaling_penalty: float,
    output: Path,
):
    config = VARIANTS[variant]

    env = make_env(
        "train",
        on_time_reward=on_time_reward,
        failed_frame_penalty=failed_frame_penalty,
        worker_cost=worker_cost,
        scaling_penalty=scaling_penalty,
    )

    check_env(
        env,
        warn=True,
    )

    model = create_dqn(
        env,
        seed=seed,
        policy=config["policy"],
        n_steps=config["n_steps"],
        batch_size=64,
    )

    model.learn(
        total_timesteps=total_timesteps,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        str(output)
    )

    metadata = {
        "model": output.name,
        "variant": variant,
        "timesteps": total_timesteps,
        "seed": seed,
        "n_steps": config["n_steps"],
        "on_time_reward": on_time_reward,
        "failed_frame_penalty": failed_frame_penalty,
        "worker_cost": worker_cost,
        "scaling_penalty": scaling_penalty,
        "batch_size": 64,
        "buffer_size": 50000,
        "learning_starts": 1000,
    }

    metadata_path = output.with_suffix(
        ".json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    env.close()

    print()
    print(
        f"Variant: {variant}"
    )

    print(
        f"Model: {output}.zip"
    )

    print(
        f"Config: {metadata_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        required=True,
    )

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
        "--on-time-reward",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--failed-frame-penalty",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--worker-cost",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--scaling-penalty",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.output is None:
        output = (
            ROOT
            / "results"
            / "models"
            / f"{args.variant}_seed{args.seed}_{args.timesteps // 1000}k"
        )
    else:
        output = args.output

    train_variant(
        variant=args.variant,
        total_timesteps=args.timesteps,
        seed=args.seed,
        on_time_reward=args.on_time_reward,
        failed_frame_penalty=args.failed_frame_penalty,
        worker_cost=args.worker_cost,
        scaling_penalty=args.scaling_penalty,
        output=output,
    )


if __name__ == "__main__":
    main()