from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulatorConfig:
    camera_fps: float = 10.0
    service_seconds: float = 0.07       # Temporary for Part 2B
    startup_seconds: float = 2.0        # Temporary for Part 2B
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

    def _advance_tick(self, camera_ids):
        now = self.tick

        # Finish frames whose processing ends now.
        for worker in self.workers[:]:
            if (worker["finish_at"] is not None and worker["finish_at"] <= now):
                camera_id, arrival_tick = worker["frame"]
                latency = now - arrival_tick

                if latency <= self.deadline_ticks:
                    self.on_time += 1
                else:
                    self.late += 1

                worker["finish_at"] = None
                worker["frame"] = None

                if worker["retiring"]:
                    self.workers.remove(worker)

        # A frame still waiting at its deadline is dropped.
        while (self.queue and now - self.queue[0][1] >= self.deadline_ticks):
            self.queue.popleft()
            self.dropped_deadline += 1

        # Each active camera sends one frame every period seconds.
        if now % self.frame_period_ticks == 0:
            for camera_id in camera_ids:
                self.arrived += 1
                if len(self.queue) < self.queue_capacity:
                    self.queue.append((camera_id, now))
                else:
                    self.dropped_full += 1

        # Free, ready workers take the oldest frames first.
        for worker in self.workers:
            if (self.queue and worker["ready_at"] <= now and worker["finish_at"] is None and not worker["retiring"]):
                worker["frame"] = self.queue.popleft()
                worker["finish_at"] = now + self.service_ticks

        # Count all allocated workers, including starting and retiring ones.
        self.worker_seconds += len(self.workers) * self.config.tick_seconds
        self.tick += 1

    def step(self, action, camera_ids):
        #Make one scaling decision, then simulate one decision interval.
        camera_ids = tuple(camera_ids)
        action_applied = self._scale(action)

        for _ in range(self.decision_ticks):
            self._advance_tick(camera_ids)

        processing = 0

        for worker in self.workers:
            if worker["finish_at"] is not None:
                processing += 1

        return {
            "time_seconds": self.tick * self.config.tick_seconds,
            "action_applied": action_applied,
            "arrived": self.arrived,
            "on_time": self.on_time,
            "late": self.late,
            "dropped_deadline": self.dropped_deadline,
            "dropped_full": self.dropped_full,
            "waiting": len(self.queue),
            "processing": processing,
            "workers": len(self.workers),
            "worker_seconds": round(self.worker_seconds, 2)}



if __name__ == "__main__":
# Small example: demand rises to three cameras, then falls to one.
    config = SimulatorConfig(
        deadline_seconds=5.0,
        decision_seconds=2.0,
        max_workers=6,
        queue_capacity_frames=100,
    )
    sim = Simulator(config)

    for second in range(10):
        cameras = range(3) if 3 <= second < 7 else range(1)

        if second in (3, 4):
            action = "add"
        elif second in (7, 8):
            action = "remove"
        else:
            action = "hold"

        result = sim.step(action, cameras)
        print(result)