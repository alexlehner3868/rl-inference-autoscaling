PRIMARY_METRICS = [
    "on_time_percentage",
    "worker_seconds",
]


SECONDARY_METRICS = [
    "arrived",
    "on_time",
    "late",
    "dropped_deadline",
    "dropped_full",
    "unfinished_frames",
    "mean_latency_seconds",
    "p95_latency_seconds",
    "p99_latency_seconds",
    "max_queue_length",
    "scaling_changes",
    "blocked_actions",
    "average_workers",
    "per_camera",
]


SHARED_METRICS = (
    PRIMARY_METRICS
    + SECONDARY_METRICS
)