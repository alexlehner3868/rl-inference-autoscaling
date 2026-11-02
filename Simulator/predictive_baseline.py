from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .Simulator_queue import Simulator, SimulatorConfig
else:
    from Simulator_queue import Simulator, SimulatorConfig


# Controller settings
WINDOW = 5 # Number of observations to calculate demand based on
SCALE_DOWN_STEPS = 3 # Require 3 demand decrease observations before removing a worker 
MAX_WORKERS = 4 # Max workers allowed 
INITIAL_WORKERS = 1 # Start with 1 worker 
DEADLINE_SECONDS = 1.0
QUEUE_SIZE = 50 # Matches fixed baseline 
ROOT = Path(__file__).resolve().parent.parent


class PredictiveController:
    def __init__(self):
        self.demand = deque(maxlen=WINDOW) # Store last 5 demand observations
        self.previous_arrived = 0
        self.low_steps = 0
        self.forecast = 0.0
        self.target = INITIAL_WORKERS

    def choose_action(self, sim):
        interval = sim.decision_ticks * sim.config.tick_seconds # Interval --> Seconds 
        if sim.tick > 0:
            # Calculate demand in last window (FPS) 
            self.demand.append((sim.arrived - self.previous_arrived) / interval)
        self.previous_arrived = sim.arrived

        # Forcase the demand using straight line estimation 
        self.forecast = self.demand[-1] if self.demand else 0.0
        if len(self.demand) >= 2:
            times = np.arange(len(self.demand)) * interval
            slope, intercept = np.polyfit(times, list(self.demand), 1)
            horizon = sim.startup_ticks * sim.config.tick_seconds
            self.forecast = max(0.0, float(intercept + slope * (times[-1] + horizon)))

        # Convert demand into required num workers 
        capacity = 1 / (sim.service_ticks * sim.config.tick_seconds)

        # Get target number of workers
        self.target = max(sim.min_serving_workers, min(sim.max_workers, math.ceil(self.forecast / capacity - 1e-9)))

        # Look at the num workers available or soon to be available 
        ready = sum(w["ready_at"] <= sim.tick and not w["retiring"] for w in sim.workers)
        starting = sum(w["ready_at"] > sim.tick and not w["retiring"] for w in sim.workers)

        # Determine action to take 
        if self.target > ready + starting and len(sim.workers) < sim.max_workers: 
            # Add if target exceeds capacity and we have available worker 
            self.low_steps = 0
            return "add"
        if self.target < ready + starting and not sim.queue and starting == 0:
            # Remove worker if demand is low, there is no queue and no workers are starting  
            self.low_steps += 1
            if self.low_steps >= SCALE_DOWN_STEPS and ready > sim.min_serving_workers:
                self.low_steps = 0
                return "remove"
        else:
            self.low_steps = 0
        return "hold"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_workload(file, output, calibration):
    # Load workload and init configurations 
    workload = json.loads(file.read_text())
    config = SimulatorConfig(
        camera_fps=workload["camera_fps"], decision_seconds=workload["decision_seconds"],
        service_seconds=calibration["service_time_mean_seconds"],
        startup_seconds=calibration["startup_mean_seconds"],
        deadline_seconds=DEADLINE_SECONDS, queue_capacity_frames=QUEUE_SIZE,
        initial_workers=INITIAL_WORKERS, max_workers=MAX_WORKERS,
    )

    # Init sim and controller
    sim = Simulator(config)
    controller = PredictiveController()

    decision = sim.decision_ticks * config.tick_seconds
    # Iterate through all the intervals in the workload
    for interval in workload["intervals"]:
        time = sim.tick * config.tick_seconds 
        steps = (interval["end_seconds"] - time) / decision

        for _ in range(round(steps)):
            # Choose action and execute it 
            action = controller.choose_action(sim)
            sim.step(action, interval["camera_ids"])
            # Save results 
            sim.history[-1].update(observed_demand_fps=controller.demand[-1] if controller.demand else 0,
                                   forecast_demand_fps=controller.forecast,
                                   target_workers=controller.target)
    
    final = sim.history[-1]

    # Save results
    sim.save_results(output / file.stem, controller="predictive", workload=file.stem)
    max_capacity = MAX_WORKERS / (sim.service_ticks * config.tick_seconds)
    result = {
        "workload": file.stem, "pattern": workload["pattern"], "load_level": workload["load_level"],
        "duration_seconds": final["time_seconds"], "arrived": sim.arrived,
        "on_time": sim.on_time, "on_time_percentage": final["on_time_percentage"],
        "late": sim.late, "dropped_deadline": sim.dropped_deadline,
        "dropped_full": sim.dropped_full,
        "unfinished_frames": final["waiting"] + final["processing"],
        "worker_seconds": round(final["worker_seconds"], 6),
        "average_workers": round(final["worker_seconds"], 6) / final["time_seconds"],
        "p95_latency_seconds": float(np.percentile(sim.latencies_seconds, 95)) if sim.latencies_seconds else None,
        "max_queue_length": sim.max_queue_length,
        "scaling_changes": sim.action_counts["add"] + sim.action_counts["remove"],
        "blocked_actions": sim.action_counts["blocked"],
        "overload_seconds": sum(row["demand_fps"] > max_capacity for row in sim.history) * decision,
        "result_dir": file.stem,
    }
    return result, sim.latencies_seconds


# Save results to a CSV 
def report_results(output, results, latencies):
    write_csv(output / "comparison.csv", results)
    overall = []
    
    for group in ("all", "standard", "stress"):
        selected = [i for i, row in enumerate(results) if group == "all" or row["load_level"] == group]
        
        if not selected:
            continue
        
        rows = [results[i] for i in selected]
        samples = [latency for i in selected for latency in latencies[i]]
        arrived = sum(row["arrived"] for row in rows)
        worker_seconds = round(sum(row["worker_seconds"] for row in rows), 6)
        duration = sum(row["duration_seconds"] for row in rows)
        overall.append({
            "group": group, "workloads": len(rows), "arrived": arrived,
            "on_time_percentage": 100 * sum(row["on_time"] for row in rows) / arrived if arrived else None,
            "mean_workload_on_time_percentage": float(np.mean([row["on_time_percentage"] for row in rows if row["on_time_percentage"] is not None])),
            "worker_seconds": worker_seconds, "average_workers": worker_seconds / duration,
            "late": sum(row["late"] for row in rows),
            "dropped_deadline": sum(row["dropped_deadline"] for row in rows),
            "dropped_full": sum(row["dropped_full"] for row in rows),
            "unfinished_frames": sum(row["unfinished_frames"] for row in rows),
            "p95_latency_seconds": float(np.percentile(samples, 95)) if samples else None,
            "scaling_changes": sum(row["scaling_changes"] for row in rows),
            "overload_seconds": sum(row["overload_seconds"] for row in rows),
        })
    
    write_csv(output / "overall.csv", overall)


def main():
    files = sorted((ROOT / "workloads" / "validation").glob("*.json"))    
    calibration = json.loads((ROOT / "results/calibration/simulator_parameters.json").read_text())
    output = ROOT / "results" / "predictive_baseline" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    results, latencies = [], []
    
    for number, file in enumerate(files, start=1):
        result, samples = run_workload(file, output, calibration)
        results.append(result)
        latencies.append(samples)
       
        print(f"{number}/{len(files)} {file.stem}: {result['on_time_percentage']:.2f}% on time, {result['worker_seconds']:.2f} worker-seconds", flush=True)
    
    report_results(output, results, latencies)
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
