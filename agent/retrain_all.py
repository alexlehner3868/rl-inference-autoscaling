"""Train and compare every controller with one reward and resource budget."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
from itertools import product
import json
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from Simulator.baseline_controllers import QueueThresholdController
from Simulator.predictive_baseline import PredictiveController, write_csv
from Simulator.Simulator_queue import SimulatorConfig
from agent.evaluate_dqn import evaluate_workload
from agent.train_dqn import ROOT, load_calibration, load_workloads
from agent.train_dqn_variants import train_variant
from agent.train_ppo import train_ppo
from agent.train_two_dqn import train_two_dqn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEEDS = (0, 1, 2, 3, 4)
TOTAL_STEPS = 61440
METHODS = ("vanilla", "nstep", "dueling", "dueling_nstep", "two_dqn", "ppo")
REWARD = dict(on_time_reward=1.0, failed_frame_penalty=2.0, worker_cost=1.0, scaling_penalty=0.25)


def totals(rows, latencies):
    arrived = sum(row["arrived"] for row in rows)
    duration = sum(row["duration_seconds"] for row in rows)
    workers = sum(row["worker_seconds"] for row in rows)
    return {
        "workloads": len(rows), "arrived": arrived,
        "on_time": sum(row["on_time"] for row in rows),
        "on_time_percentage": 100 * sum(row["on_time"] for row in rows) / arrived,
        "total_reward": sum(row["total_reward"] for row in rows),
        "worker_seconds": workers, "average_workers": workers / duration,
        "peak_workers": max(row["peak_workers"] for row in rows),
        "late": sum(row["late"] for row in rows),
        "dropped_frames": sum(row["dropped_deadline"] + row["dropped_full"] for row in rows),
        "unfinished_frames": sum(row["unfinished_frames"] for row in rows),
        "scaling_changes": sum(row["scaling_changes"] for row in rows),
        "p95_latency_seconds": float(np.percentile(latencies, 95)) if latencies else None,
    }


def evaluate(model, *, settings=None, output=None, controller="dqn"):
    rows, latencies = [], []
    calibration = load_calibration()
    for workload in load_workloads("validation"):
        options = {}
        if settings is not None:
            method = settings["method"]
            if method == "fixed":
                options = dict(action_selector=lambda sim, cameras: "hold",
                               initial_workers=settings["workers"])
            elif method == "predictive":
                policy = PredictiveController(settings["window"], settings["scale_down_steps"])
                options["action_selector"] = lambda sim, cameras: policy.choose_action(sim)
            else:
                policy = QueueThresholdController(settings["add_threshold"], settings["remove_threshold"],
                                                  settings["scale_down_steps"])
                options["action_selector"] = policy.choose_action
        row = evaluate_workload(model, workload, calibration, output=output, controller=controller, **options)
        latencies.extend(row.pop("latencies"))
        rows.append(row)
    overall = totals(rows, latencies)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "comparison.csv", rows)
        (output / "overall.json").write_text(json.dumps(overall, indent=2) + "\n")
    return overall, rows


def train_model(task):
    output, method, seed = task
    torch.set_num_threads(1)
    name = f"{method}_seed{seed}"
    model_path = output / "models" / f"{name}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with (output / f"{name}.log").open("w") as log, redirect_stdout(log):
        if method == "two_dqn":
            train_two_dqn(TOTAL_STEPS, seed, model_path, worker_cost=REWARD["worker_cost"])
        elif method == "ppo":
            train_ppo(TOTAL_STEPS, seed, model_path)
        else:
            train_variant(method, TOTAL_STEPS, seed, output=model_path, **REWARD)
    algorithm = PPO if method == "ppo" else DQN
    model = algorithm.load(model_path)
    actual_steps = model.num_timesteps * (2 if method == "two_dqn" else 1)
    if actual_steps != TOTAL_STEPS:
        raise ValueError(f"Unexpected training budget for {name}: {actual_steps}")
    metadata_path = model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(method=method, seed=seed, total_timesteps=actual_steps, reward=REWARD,
                    max_workers=SimulatorConfig.max_workers, simulator_configuration=vars(SimulatorConfig()))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    overall, rows = evaluate(model)
    result = dict(method=method, seed=seed, model_path=str(model_path.relative_to(ROOT)), **overall)
    folder = output / "validation" / name
    folder.mkdir(parents=True)
    write_csv(folder / "comparison.csv", rows)
    (folder / "overall.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def tune_baseline(task):
    output, method = task
    torch.set_num_threads(1)
    if method == "fixed":
        settings = [dict(method=method, workers=n) for n in range(1, SimulatorConfig.max_workers + 1)]
    elif method == "predictive":
        settings = [dict(method=method, window=w, scale_down_steps=d)
                    for w, d in product((3, 5, 10), (1, 3, 5))]
    else:
        settings = [dict(method=method, add_threshold=a, remove_threshold=r, scale_down_steps=d)
                    for a, r, d in product((3, 5, 10), (0, 1, 2), (1, 3, 5, 10))]
    rows = []
    for setting in settings:
        overall, _ = evaluate(None, settings=setting)
        rows.append({**setting, **overall})
    folder = output / "baselines" / method
    folder.mkdir(parents=True)
    write_csv(folder / "tuning.csv", rows)
    # Use the shared reward to select one configuration across all workloads.
    best = max(rows, key=lambda row: (row["total_reward"], -row["worker_seconds"]))
    chosen = {key: best[key] for key in settings[0]}
    (folder / "settings.json").write_text(json.dumps(chosen, indent=2) + "\n")
    overall, _ = evaluate(None, settings=chosen, output=folder / "validation", controller=method)
    return dict(method=method, seed="baseline", model_path=str((folder / "settings.json").relative_to(ROOT)), **overall)


def report(output, results):
    write_csv(output / "all_runs.csv", results)
    groups, selected = [], []
    for method in (*METHODS, "fixed", "predictive", "queue"):
        rows = [row for row in results if row["method"] == method]
        if not rows:
            continue
        row = dict(method=method, runs=len(rows))
        for metric in ("total_reward", "on_time_percentage", "worker_seconds", "peak_workers"):
            values = [r[metric] for r in rows]
            row[metric + "_mean"] = mean(values)
            row[metric + "_std"] = stdev(values) if len(values) > 1 else 0.0
        groups.append(row)
        selected.append(max(rows, key=lambda r: r["total_reward"]))
    write_csv(output / "method_summary.csv", groups)
    write_csv(output / "selected_models.csv", selected)
    return selected


def plot_results(output, results, selected):
    methods = (*METHODS, "fixed", "predictive", "queue")
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for index, method in enumerate(methods):
        rows = [row for row in results if row["method"] == method]
        ax.scatter([float(row["worker_seconds"]) for row in rows],
                   [float(row["on_time_percentage"]) for row in rows],
                   color=colors(index), alpha=0.6, label=method)
        best = next(row for row in selected if row["method"] == method)
        ax.scatter(float(best["worker_seconds"]), float(best["on_time_percentage"]),
                   color=colors(index), marker="*", s=160, edgecolor="black", linewidth=0.5)
    ax.set(xlabel="Allocated worker-seconds", ylabel="Frames completed on time (%)",
           title="Shared reward and 10-worker limit — validation results")
    ax.legend(fontsize=8, loc="lower right")
    ax.text(0.02, 0.02, "Stars mark validation-selected checkpoints or baseline settings", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    rewards, deviations = [], []
    for method in methods:
        values = [float(row["total_reward"]) for row in results if row["method"] == method]
        rewards.append(mean(values))
        deviations.append(stdev(values) if len(values) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    ax.barh(methods, rewards, xerr=deviations, capsize=4,
            color=[colors(index) for index in range(len(methods))])
    ax.set(xlabel="Validation total reward — mean ± sample standard deviation",
           title="Shared objective — five seeds per RL method; tuned baselines")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(output / "reward_comparison.png", dpi=160)
    plt.close(fig)


def main():
    output = ROOT / "results" / "shared_objective" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps({
        "reward": REWARD, "max_workers": SimulatorConfig.max_workers,
        "total_steps_per_run": TOTAL_STEPS, "seeds": SEEDS,
        "training_split": "train", "selection_split": "validation",
        "selection_metric": "total_reward", "test_workloads_used": False,
        "observation_order": ["queue_length", "oldest_wait_seconds", "demand_fps",
                              "serving_workers", "starting_workers"],
        "actions": {0: "remove", 1: "hold", 2: "add"},
    }, indent=2) + "\n")
    tasks = [(output, method, seed) for method, seed in product(METHODS, SEEDS)]
    results = []
    print(f"Training {len(tasks)} models and tuning three baselines. Results: {output}", flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(tune_baseline, (output, method)) for method in ("fixed", "predictive", "queue")]
        futures.extend(pool.submit(train_model, task) for task in tasks)
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            report(output, results)
            print(f"{len(results)}/33 {result['method']} seed {result['seed']}: "
                  f"reward {result['total_reward']:,.2f}, {result['on_time_percentage']:.2f}% on time", flush=True)
    selected = report(output, results)
    for row in selected:
        if row["method"] not in METHODS:
            continue
        algorithm = PPO if row["method"] == "ppo" else DQN
        model = algorithm.load(ROOT / row["model_path"])
        evaluate(model, output=output / "selected" / row["method"], controller=row["method"])
    plot_results(output, results, selected)
    print(f"Finished. Share paths in {output / 'selected_models.csv'}", flush=True)


if __name__ == "__main__":
    main()
