"""Queue controller shared by standalone runs and validation tuning."""


class QueueThresholdController:
    def __init__(self, add_threshold=3, remove_threshold=1, scale_down_steps=10):
        self.add_threshold = add_threshold
        self.remove_threshold = remove_threshold
        self.scale_down_steps = scale_down_steps
        self.low_steps = 0

    def choose_action(self, sim, camera_ids):
        waiting, workers = len(sim.queue), len(sim.workers)
        if waiting >= self.add_threshold and workers < sim.max_workers:
            self.low_steps = 0
            return "add"
        if waiting <= self.remove_threshold and workers > sim.min_serving_workers:
            self.low_steps += 1
            all_ready = all(worker["ready_at"] <= sim.tick and not worker["retiring"]
                            for worker in sim.workers)
            demand = len(camera_ids) * sim.config.camera_fps
            capacity_after_remove = (workers - 1) / sim.config.service_seconds
            if self.low_steps >= self.scale_down_steps and all_ready and demand <= capacity_after_remove:
                self.low_steps = 0
                return "remove"
        else:
            self.low_steps = 0
        return "hold"
