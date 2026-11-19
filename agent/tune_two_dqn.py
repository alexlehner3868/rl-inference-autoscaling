"""Compare worker costs and training budgets using a common validation reward."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
import json
from itertools import product
from multiprocessing import get_context
from statistics import mean, stdev

import matplotlib
import torch
from stable_baselines3 import DQN

from Simulator.predictive_baseline import write_csv
from Simulator.Simulator_queue import calculate_reward
from agent.evaluate_dqn import evaluate_workload
from agent.train_dqn import ROOT, load_calibration, load_workloads
from agent.train_two_dqn import train_two_dqn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKER_COSTS = (0.1, 0.5, 1.0, 2.0)
STEPS_PER_NETWORK = (15000, 30000)
SEEDS = (0, 1, 2, 3, 4)


def summarize(settings, rows):
    arrived = sum(row["arrived"] for row in rows)
    return {
        **settings,
        "on_time_percentage": 100 * sum(row["on_time"] for row in rows) / arrived,
        "worker_seconds": sum(row["worker_seconds"] for row in rows),
        "average_workers": sum(row["worker_seconds"] for row in rows) / sum(row["duration_seconds"] for row in rows),
        "common_total_reward": sum(row["total_reward"] for row in rows),
        "training_weight_reward": sum(row["training_weight_reward"] for row in rows),
        "dropped_frames": sum(row["dropped_deadline"] + row["dropped_full"] for row in rows),
        "late": sum(row["late"] for row in rows),
        "scaling_changes": sum(row["scaling_changes"] for row in rows),
        "peak_workers": max(row["peak_workers"] for row in rows),
        "zero_demand_additions": sum(row["zero_demand_additions"] for row in rows),
    }


def aggregate_seeds(summary):
    grouped = []
    metrics = ("on_time_percentage", "worker_seconds", "common_total_reward",
               "average_workers", "peak_workers", "dropped_frames", "scaling_changes",
               "zero_demand_additions")
    for steps, cost in product(STEPS_PER_NETWORK, WORKER_COSTS):
        rows = [row for row in summary if row["steps_per_network"] == steps and row["worker_cost"] == cost]
        if not rows:
            continue
        result = {"worker_cost": cost, "steps_per_network": steps, "seeds": len(rows)}
        for metric in metrics:
            values = [row[metric] for row in rows]
            result[metric + "_mean"] = mean(values)
            result[metric + "_std"] = stdev(values) if len(values) > 1 else 0.0
        result["peak_workers_max"] = max(row["peak_workers"] for row in rows)
        grouped.append(result)
    return grouped


def plot_results(output, grouped):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    metrics = (
        ("on_time_percentage", "Frames completed on time (%)"),
        ("worker_seconds", "Allocated worker-seconds"),
        ("common_total_reward", "Common total reward (worker cost 1.0)"),
    )
    for ax, (metric, title) in zip(axes, metrics):
        for steps, marker in zip(STEPS_PER_NETWORK, ("o", "s")):
            rows = [row for row in grouped if row["steps_per_network"] == steps]
            ax.errorbar([row["worker_cost"] for row in rows],
                        [row[metric + "_mean"] for row in rows],
                        yerr=[row[metric + "_std"] for row in rows],
                        marker=marker, capsize=4, label=f"{steps:,} steps per network")
        ax.set(title=title, xlabel="Training worker cost")
        ax.set_xticks(WORKER_COSTS)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Two-network DQN — {len(SEEDS)} seeds, mean ± sample standard deviation")
    fig.savefig(output / "seed_comparison.png", dpi=160)
    plt.close(fig)


def run_configuration(task):
    output, steps, cost, seed = task
    # Each process has its own random state, models, environments and replay buffers.
    torch.set_num_threads(1)
    name = f"cost_{cost:g}_steps_{steps}_seed_{seed}"
    model_path = output / "models" / f"{name}.zip"
    settings = {"worker_cost": cost, "steps_per_network": steps, "seed": seed}
    with (output / f"{name}.log").open("w") as log, redirect_stdout(log):
        train_two_dqn(2 * steps, seed, model_path, worker_cost=cost)
    model = DQN.load(model_path)
    rows = []
    calibration = load_calibration()
    for workload in load_workloads("validation"):
        # Use the agreed common objective for comparison across training costs.
        row = evaluate_workload(model, workload, calibration)
        row.pop("latencies")
        row.update(settings)
        row["training_weight_reward"] = calculate_reward(
            row["on_time"], row["late"] + row["dropped_deadline"] + row["dropped_full"],
            row["worker_seconds"], row["scaling_changes"], worker_cost=cost)
        rows.append(row)
    result = summarize(settings, rows)
    # Preserve each completed run even if the overall sweep is interrupted.
    (output / f"{name}.json").write_text(json.dumps({"summary": result, "workloads": rows}, indent=2) + "\n")
    return result, rows


def main():
    output = ROOT / "results" / "two_dqn_tuning" / datetime.now().strftime("multiseed_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output / "settings.json").write_text(json.dumps({
        "worker_costs": WORKER_COSTS, "steps_per_network": STEPS_PER_NETWORK, "seeds": SEEDS,
        "evaluation_reward": {"on_time_reward": 1.0, "failed_frame_penalty": 2.0,
                              "worker_cost": 1.0, "scaling_penalty": 0.25},
        "training_split": "train", "evaluation_split": "validation", "max_workers": None,
    }, indent=2) + "\n")
    tasks = [(output, steps, cost, seed) for steps, cost, seed in product(STEPS_PER_NETWORK, WORKER_COSTS, SEEDS)]
    summary, per_workload = [], []
    print(f"Running {len(tasks)} experiments. Results: {output}", flush=True)
    # Independent processes allow four models to train without sharing random state.
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_configuration, task) for task in tasks]
        for future in as_completed(futures):
            result, rows = future.result()
            summary.append(result)
            per_workload.extend(rows)
            summary.sort(key=lambda row: (row["steps_per_network"], row["worker_cost"], row["seed"]))
            per_workload.sort(key=lambda row: (row["steps_per_network"], row["worker_cost"], row["seed"], row["workload"]))
            write_csv(output / "summary.csv", summary)
            write_csv(output / "per_workload.csv", per_workload)
            write_csv(output / "seed_summary.csv", aggregate_seeds(summary))
            print(f"{len(summary)}/{len(tasks)} cost {result['worker_cost']:g}, "
                  f"steps {result['steps_per_network']}, seed {result['seed']}: "
                  f"{result['on_time_percentage']:.2f}% on time, "
                  f"peak {result['peak_workers']}, reward {result['common_total_reward']:,.2f}", flush=True)
    plot_results(output, aggregate_seeds(summary))
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
