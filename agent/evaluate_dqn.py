import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from agent.dqn_env import InferenceScalingEnv
from agent.train_dqn import load_calibration, load_workloads
from agent.dueling_dqn import DuelingDQNPolicy


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "results" / "models" / "dqn_autoscaler_30k.zip"


def evaluate_workload(model, workload, calibration):
    env = InferenceScalingEnv(
        workloads=[workload],
        calibration=calibration,
    )

    observation, _ = env.reset(seed=0)
    terminated = False

    while not terminated:
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, _, terminated, _, _ = env.step(
            int(action)
        )

    sim = env.simulator
    final = sim.history[-1]

    latencies = sim.latencies_seconds

    return {
        "workload": workload["name"],
        "pattern": workload["pattern"],
        "load_level": workload["load_level"],
        "arrived": sim.arrived,
        "on_time": sim.on_time,
        "on_time_percentage": (
            100 * sim.on_time / sim.arrived
            if sim.arrived
            else 0
        ),
        "late": sim.late,
        "dropped_deadline": sim.dropped_deadline,
        "dropped_full": sim.dropped_full,
        "unfinished_frames": (
            final["waiting"] + final["processing"]
        ),
        "worker_seconds": sim.worker_seconds,
        "average_workers": (
            sim.worker_seconds / final["time_seconds"]
        ),
        "p95_latency_seconds": (
            float(np.percentile(latencies, 95))
            if latencies
            else None
        ),
        "max_queue_length": sim.max_queue_length,
        "scaling_changes": (
            sim.action_counts["add"]
            + sim.action_counts["remove"]
        ),
        "blocked_actions": sim.action_counts["blocked"],
        "duration_seconds": final["time_seconds"],
        "latencies": latencies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    model = DQN.load(args.model)

    workloads = load_workloads("validation")
    calibration = load_calibration()

    results = []
    all_latencies = []

    for workload in workloads:
        result = evaluate_workload(
            model,
            workload,
            calibration,
        )

        all_latencies.extend(result.pop("latencies"))
        results.append(result)

        print(
            f"{result['workload']}: "
            f"{result['on_time_percentage']:.2f}% on time, "
            f"{result['late']} late, "
            f"{result['dropped_deadline'] + result['dropped_full']} dropped, "
            f"{result['worker_seconds']:.1f} worker-seconds"
        )

    total_arrived = sum(
        result["arrived"]
        for result in results
    )

    total_on_time = sum(
        result["on_time"]
        for result in results
    )

    total_duration = sum(
        result["duration_seconds"]
        for result in results
    )

    total_worker_seconds = sum(
        result["worker_seconds"]
        for result in results
    )

    overall = {
        "model": args.model.name,
        "workloads": len(results),
        "arrived": total_arrived,
        "on_time": total_on_time,
        "on_time_percentage": (
            100 * total_on_time / total_arrived
            if total_arrived
            else 0
        ),
        "late": sum(
            result["late"]
            for result in results
        ),
        "dropped_deadline": sum(
            result["dropped_deadline"]
            for result in results
        ),
        "dropped_full": sum(
            result["dropped_full"]
            for result in results
        ),
        "unfinished_frames": sum(
            result["unfinished_frames"]
            for result in results
        ),
        "worker_seconds": total_worker_seconds,
        "average_workers": (
            total_worker_seconds / total_duration
        ),
        "p95_latency_seconds": (
            float(np.percentile(all_latencies, 95))
            if all_latencies
            else None
        ),
        "scaling_changes": sum(
            result["scaling_changes"]
            for result in results
        ),
        "blocked_actions": sum(
            result["blocked_actions"]
            for result in results
        ),
    }

    output = (
        ROOT
        / "results"
        / "dqn_validation"
        / f"{args.model.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output / "comparison.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    with open(
        output / "overall.json",
        "w",
    ) as f:
        json.dump(
            overall,
            f,
            indent=2,
        )

    print()
    print(f"DQN VALIDATION RESULTS - {args.model.stem}")
    print(
        f"On-time: "
        f"{overall['on_time_percentage']:.2f}%"
    )
    print(
        f"Late frames: {overall['late']}"
    )
    print(
        f"Dropped frames: "
        f"{overall['dropped_deadline'] + overall['dropped_full']}"
    )
    print(
        f"Worker-seconds: "
        f"{overall['worker_seconds']:.1f}"
    )
    print(
        f"Average workers: "
        f"{overall['average_workers']:.2f}"
    )
    print(
        f"P95 latency: "
        f"{overall['p95_latency_seconds']:.3f} s"
    )
    print(
        f"Scaling changes: "
        f"{overall['scaling_changes']}"
    )
    print(
        f"Blocked actions: "
        f"{overall['blocked_actions']}"
    )

    print()
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
