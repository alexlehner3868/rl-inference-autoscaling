from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import json


@dataclass(frozen=True)
class SimulatorConfig:
    camera_fps: float = 10.0
    
    service_seconds: float = 0.07197132544888889 # Measured CPU means from profiling
    startup_seconds: float = 2.3057613082 # Measured CPU means from profiling
    deadline_seconds: float = 1.0
    decision_seconds: float = 1.0
    tick_seconds: float = 0.001

    initial_workers: int = 1
    min_serving_workers: int = 1
    max_workers: int = 4                # Temporary
    queue_capacity_frames: int = 50    # Temporary



class Simulator:
    def __init__(self, config=None):
        self.config = config or SimulatorConfig()


        if not (1 <= self.config.min_serving_workers <= self.config.initial_workers <= self.config.max_workers):
            raise ValueError("Invalid worker limits")

        if min(
            self.config.camera_fps,
            self.config.service_seconds,
            self.config.startup_seconds,
            self.config.deadline_seconds,
            self.config.decision_seconds,
            self.config.tick_seconds,
            self.config.queue_capacity_frames) <= 0:

            raise ValueError("Times, frame rate, and queue capacity must be positive")

        def ticks(seconds):
            return max(1, round(seconds / self.config.tick_seconds))

        self.max_workers = self.config.max_workers
        self.min_serving_workers = self.config.min_serving_workers
        self.queue_capacity = self.config.queue_capacity_frames

        self.frame_period_ticks = ticks(1 / self.config.camera_fps)
        self.service_ticks = ticks(self.config.service_seconds)
        self.startup_ticks = ticks(self.config.startup_seconds)
        self.deadline_ticks = ticks(self.config.deadline_seconds)
        self.decision_ticks = ticks(self.config.decision_seconds)

        self.tick = 0
        self.queue = deque()
        self.workers = [self._new_worker(ready_at=0) for _ in range(self.config.initial_workers)]

        self.arrived = 0
        self.on_time = 0
        self.late = 0
        self.dropped_deadline = 0
        self.dropped_full = 0
        self.worker_seconds = 0.0

        # Stats for final simulator performance reporting
        self.history = []
        self.capacity_timeline = []
        self.latencies_seconds = []
        self.camera_stats = {}
        self.max_queue_length = 0
        self.action_counts = {"add": 0, "remove": 0, "hold": 0, "blocked": 0}

    # define a New worker with ready_at, finish_at, frame, and retiring attributes
    def _new_worker(self, ready_at):
        return {"ready_at": ready_at,"finish_at": None,"frame": None,"retiring": False}
    
    # Defines what happens with each action: hold, add, or remove workers. Returns True if the action was applied, False otherwise.
    def _scale(self, action):
        if action == "hold":
            return True

        if action == "add":
            # Starting and retiring workers still occupy allocations.
            if len(self.workers) >= self.max_workers:
                return False
            self.workers.append(self._new_worker(self.tick + self.startup_ticks))
            return True

        if action == "remove":
            # Do not cancel a starting worker or remove the last worker that can serve frames.
            serving = []
            for worker in self.workers:
                if (worker["ready_at"] <= self.tick and not worker["retiring"]):
                    serving.append(worker)
            if len(serving) <= self.min_serving_workers:
                return False


            # Remove an idle worker first.
            idle_worker = None

            for worker in serving:
                if worker["finish_at"] is None:
                    idle_worker = worker
                    break

            if idle_worker is not None:
                self.workers.remove(idle_worker)
            else:
                # Everyone is busy. The last worker finishes its frame, then leaves.
                serving[-1]["retiring"] = True

            return True

        raise ValueError("Action must be 'add', 'hold', or 'remove'")

    def _serving_workers(self):
        return sum(worker["ready_at"] <= self.tick and not worker["retiring"]
                   for worker in self.workers)

    def _advance_tick(self, camera_ids):
        now = self.tick

        # Finish frames whose processing ends now.
        for worker in self.workers[:]:
            if (worker["finish_at"] is not None and worker["finish_at"] <= now):
                camera_id, arrival_tick = worker["frame"]
                latency = now - arrival_tick
                self.latencies_seconds.append(latency * self.config.tick_seconds)

                if latency <= self.deadline_ticks:
                    self.on_time += 1
                    self.camera_stats[camera_id]["on_time"] += 1
                else:
                    self.late += 1
                    self.camera_stats[camera_id]["late"] += 1

                worker["finish_at"] = None
                worker["frame"] = None

                if worker["retiring"]:
                    self.workers.remove(worker)

        # A frame still waiting at its deadline is dropped.
        while (self.queue and now - self.queue[0][1] >= self.deadline_ticks):
            camera_id, _ = self.queue.popleft()
            self.dropped_deadline += 1
            self.camera_stats[camera_id]["dropped_deadline"] += 1

        # Each active camera sends one frame every period seconds.
        if now % self.frame_period_ticks == 0:
            for camera_id in camera_ids:
                camera = self.camera_stats.setdefault(camera_id, {
                    "arrived": 0, "on_time": 0, "late": 0,
                    "dropped_deadline": 0, "dropped_full": 0,
                })
                self.arrived += 1
                camera["arrived"] += 1
                if len(self.queue) < self.queue_capacity:
                    self.queue.append((camera_id, now))
                else:
                    self.dropped_full += 1
                    camera["dropped_full"] += 1

        # Free, ready workers take the oldest frames first.
        for worker in self.workers:
            if (self.queue and worker["ready_at"] <= now and worker["finish_at"] is None and not worker["retiring"]):
                worker["frame"] = self.queue.popleft()
                worker["finish_at"] = now + self.service_ticks

        # Record capacity changes 
        capacity = self._serving_workers() / (self.service_ticks * self.config.tick_seconds)
        if not self.capacity_timeline or capacity != self.capacity_timeline[-1][1]:
            self.capacity_timeline.append((now * self.config.tick_seconds, capacity))
        self.max_queue_length = max(self.max_queue_length, len(self.queue))

        # Count all allocated workers, including starting and retiring ones.
        self.worker_seconds += len(self.workers) * self.config.tick_seconds
        self.tick += 1

    def step(self, action, camera_ids):
        #Make one scaling decision, then simulate one decision interval.
        camera_ids = tuple(camera_ids)

        start_seconds = self.tick * self.config.tick_seconds
        completed_before = self.on_time + self.late

        action_applied = self._scale(action)

        self.action_counts[action if action_applied else "blocked"] += 1

        for _ in range(self.decision_ticks):
            self._advance_tick(camera_ids)

        processing = 0

        for worker in self.workers:
            if worker["finish_at"] is not None:
                processing += 1

        interval_seconds = self.decision_ticks * self.config.tick_seconds
        serving = self._serving_workers()
        result = {
            "interval_start_seconds": start_seconds,
            "time_seconds": self.tick * self.config.tick_seconds,
            "action": action,
            "action_applied": action_applied,
            "active_cameras": len(camera_ids),
            "demand_fps": len(camera_ids) / (self.frame_period_ticks * self.config.tick_seconds),
            "capacity_fps": serving / (self.service_ticks * self.config.tick_seconds),
            "completed_fps": (self.on_time + self.late - completed_before) / interval_seconds,
            "arrived": self.arrived,
            "on_time": self.on_time,
            "late": self.late,
            "dropped_deadline": self.dropped_deadline,
            "dropped_full": self.dropped_full,
            "waiting": len(self.queue),
            "processing": processing,
            "workers": len(self.workers),
            "serving_workers": serving,
            "starting_workers": sum(w["ready_at"] > self.tick for w in self.workers),
            "retiring_workers": sum(w["retiring"] for w in self.workers),
            "on_time_percentage": 100 * self.on_time / self.arrived if self.arrived else None,
            "worker_seconds": round(self.worker_seconds, 9)}
        self.history.append(result.copy())
        return result

    def save_results(self, output, controller="manual_example", workload="one_three_one_cameras"):
        """Save one summary, one row per decision, and a demand/capacity graph."""
        if not self.history:
            raise ValueError("Run at least one step before saving results")
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        final = self.history[-1]
        per_camera = {}
        for camera_id, counts in self.camera_stats.items():
            per_camera[str(camera_id)] = {
                **counts,
                "on_time_percentage": 100 * counts["on_time"] / counts["arrived"],
                "waiting": sum(frame[0] == camera_id for frame in self.queue),
                "processing": sum(w["frame"] is not None and w["frame"][0] == camera_id
                                  for w in self.workers),
            }
        latencies = self.latencies_seconds
        latency_stats = {
            "count": len(latencies),
            "mean": float(np.mean(latencies)) if latencies else None,
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "p99": float(np.percentile(latencies, 99)) if latencies else None,
            "max": max(latencies) if latencies else None,
        }
        scaling_changes = self.action_counts["add"] + self.action_counts["remove"]
        summary = {
            "controller": controller, "workload": workload,
            "configuration": asdict(self.config), "duration_seconds": final["time_seconds"],
            "effective_service_seconds": self.service_ticks * self.config.tick_seconds,
            "effective_startup_seconds": self.startup_ticks * self.config.tick_seconds,
            "arrived": self.arrived, "on_time": self.on_time, "late": self.late,
            "on_time_percentage": final["on_time_percentage"],
            "dropped_deadline": self.dropped_deadline, "dropped_full": self.dropped_full,
            "dropped_percentage": 100 * (self.dropped_deadline + self.dropped_full) / self.arrived
                                  if self.arrived else None,
            "waiting": final["waiting"], "processing": final["processing"],
            "unfinished_frames": final["waiting"] + final["processing"],
            "worker_seconds": round(self.worker_seconds, 9), "max_queue_length": self.max_queue_length,
            "completed_latency_seconds": latency_stats, "per_camera": per_camera,
            "action_counts": self.action_counts, "scaling_changes": scaling_changes,
            "scaling_changes_per_minute": scaling_changes * 60 / final["time_seconds"],
            "capacity_timeline": [{"time_seconds": t, "capacity_fps": rate}
                                  for t, rate in self.capacity_timeline],
            "notes": ["On-time percentage uses all arrived frames, including drops and unfinished work.",
                      "Latency statistics include completed frames only. No drain period is added.",
                      "Capacity uses ready, non-retiring workers and the tick-rounded service time.",
                      "Starting workers consume worker-seconds but provide no serving capacity."],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        with (output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(final))
            writer.writeheader()
            writer.writerows(self.history)

        # Demand applies from each interval's START; capacity follows actual readiness events.
        end = final["time_seconds"]
        demand_times = [row["interval_start_seconds"] for row in self.history] + [end]
        demand = [row["demand_fps"] for row in self.history] + [final["demand_fps"]]
        capacity_times, capacities = zip(*self.capacity_timeline)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.step(demand_times, demand, where="post", label="Camera demand", color="#d65f35", linewidth=2)
        ax.step([*capacity_times, end], [*capacities, capacities[-1]], where="post",
                label="Ready worker capacity", color="#277da1", linewidth=2)
        ax.set(title=f"Demand vs capacity — {controller}", xlabel="Simulated time (seconds)",
               ylabel="Frames per second", xlim=(0, end), ylim=(0, None))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "demand_vs_capacity.png", dpi=180)
        plt.close(fig)
        return output


if __name__ == "__main__":
# Small example: demand rises to three cameras, then falls to one.
    config = SimulatorConfig(
        deadline_seconds=5.0,
        decision_seconds=2.0,
        max_workers=6,
        queue_capacity_frames=100,
    )
    sim = Simulator(config)

    for interval in range(10):
        cameras = range(3) if 3 <= interval < 7 else range(1)

        if interval in (3, 4):
            action = "add"
        elif interval in (7, 8):
            action = "remove"
        else:
            action = "hold"

        result = sim.step(action, cameras)
        print(result)

    # Save stats
    output = Path(__file__).resolve().parent.parent / "results" / "simulations" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    print(f"Saved simulation results to {sim.save_results(output)}")
