PRIMARY_METRICS = [
    "sla_success_rate",
    "worker_seconds",
]


SECONDARY_METRICS = [
    "total_frames",
    "on_time_frames",
    "late_frames",
    "dropped_frames",
    "mean_latency",
    "p95_latency",
    "p99_latency",
    "scale_up_actions",
    "scale_down_actions",
    "average_active_workers",
    "peak_active_workers",
    "per_camera_sla",
]


SHARED_METRICS = PRIMARY_METRICS + SECONDARY_METRICS