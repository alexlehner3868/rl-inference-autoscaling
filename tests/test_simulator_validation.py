import json
from pathlib import Path

import pytest

from Simulator.Simulator_queue import (
    Simulator,
    SimulatorConfig,
)


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = (
    ROOT
    / "results"
    / "calibration"
    / "simulator_parameters.json"
)


def frame_accounting(sim):
    waiting = len(sim.queue)

    processing = sum(
        worker["frame"] is not None
        for worker in sim.workers
    )

    accounted = (
        sim.on_time
        + sim.late
        + sim.dropped_deadline
        + sim.dropped_full
        + waiting
        + processing
    )

    return accounted


def test_default_service_and_startup_match_calibration():
    calibration = json.loads(
        CALIBRATION_FILE.read_text()
    )

    config = SimulatorConfig()

    assert config.service_seconds == pytest.approx(
        calibration[
            "service_time_mean_seconds"
        ]
    )

    assert config.startup_seconds == pytest.approx(
        calibration[
            "startup_mean_seconds"
        ]
    )


def test_frame_accounting_is_conserved():
    config = SimulatorConfig(
        camera_fps=10.0,
        service_seconds=0.1,
        startup_seconds=0.5,
        deadline_seconds=2.0,
        decision_seconds=0.5,
        initial_workers=1,
        max_workers=3,
        queue_capacity_frames=20,
    )

    sim = Simulator(config)

    sequence = [
        ("hold", [0]),
        ("add", [0, 1]),
        ("hold", [0, 1]),
        ("hold", [0, 1]),
        ("remove", [0]),
        ("hold", [0]),
        ("hold", []),
    ]

    for action, cameras in sequence:
        sim.step(
            action,
            cameras,
        )

        assert sim.arrived == frame_accounting(
            sim
        )


def test_queue_stays_small_when_capacity_exceeds_demand():
    config = SimulatorConfig(
        camera_fps=5.0,
        service_seconds=0.05,
        startup_seconds=0.5,
        deadline_seconds=1.0,
        decision_seconds=0.5,
        initial_workers=1,
        max_workers=1,
        queue_capacity_frames=20,
    )

    sim = Simulator(config)

    for _ in range(20):
        sim.step(
            "hold",
            [0],
        )

    assert sim.max_queue_length <= 1
    assert sim.dropped_deadline == 0
    assert sim.dropped_full == 0
    assert sim.late == 0


def test_queue_builds_when_demand_exceeds_capacity():
    config = SimulatorConfig(
        camera_fps=10.0,
        service_seconds=0.2,
        startup_seconds=0.5,
        deadline_seconds=10.0,
        decision_seconds=1.0,
        initial_workers=1,
        max_workers=1,
        queue_capacity_frames=100,
    )

    sim = Simulator(config)

    for _ in range(5):
        sim.step(
            "hold",
            [0],
        )

    assert sim.max_queue_length > 0
    assert len(sim.queue) > 0


def test_added_worker_respects_startup_delay():
    config = SimulatorConfig(
        camera_fps=1.0,
        service_seconds=0.1,
        startup_seconds=2.0,
        deadline_seconds=5.0,
        decision_seconds=1.0,
        initial_workers=1,
        max_workers=2,
        queue_capacity_frames=20,
    )

    sim = Simulator(config)

    first = sim.step(
        "add",
        [],
    )

    assert first[
        "action_applied"
    ] is True

    assert first["workers"] == 2
    assert first["serving_workers"] == 1
    assert first["starting_workers"] == 1

    second = sim.step(
        "hold",
        [],
    )

    assert second["workers"] == 2
    assert second["serving_workers"] == 2
    assert second["starting_workers"] == 0


def test_idle_worker_is_removed_immediately():
    config = SimulatorConfig(
        camera_fps=1.0,
        service_seconds=0.1,
        startup_seconds=0.5,
        deadline_seconds=5.0,
        decision_seconds=0.5,
        initial_workers=2,
        max_workers=2,
        queue_capacity_frames=20,
    )

    sim = Simulator(config)

    result = sim.step(
        "remove",
        [],
    )

    assert result[
        "action_applied"
    ] is True

    assert result["workers"] == 1
    assert result["serving_workers"] == 1
    assert result["retiring_workers"] == 0


def test_busy_worker_finishes_before_removal():
    config = SimulatorConfig(
        camera_fps=10.0,
        service_seconds=1.0,
        startup_seconds=0.5,
        deadline_seconds=5.0,
        decision_seconds=0.1,
        initial_workers=2,
        max_workers=2,
        queue_capacity_frames=20,
    )

    sim = Simulator(config)

    sim.step(
        "hold",
        [0, 1],
    )

    result = sim.step(
        "remove",
        [],
    )

    assert result[
        "action_applied"
    ] is True

    assert result["workers"] == 2
    assert result["retiring_workers"] == 1
    assert result["serving_workers"] == 1

    for _ in range(9):
        result = sim.step(
            "hold",
            [],
        )

    assert result["workers"] == 1
    assert result["retiring_workers"] == 0


def test_starting_workers_count_toward_worker_seconds():
    config = SimulatorConfig(
        camera_fps=1.0,
        service_seconds=0.1,
        startup_seconds=2.0,
        deadline_seconds=5.0,
        decision_seconds=1.0,
        initial_workers=1,
        max_workers=2,
        queue_capacity_frames=20,
    )

    sim = Simulator(config)

    sim.step(
        "add",
        [],
    )

    assert sim.worker_seconds == pytest.approx(
        2.0,
        abs=1e-6,
    )


def test_full_queue_drops_frames_without_losing_accounting():
    config = SimulatorConfig(
        camera_fps=100.0,
        service_seconds=1.0,
        startup_seconds=0.5,
        deadline_seconds=5.0,
        decision_seconds=0.1,
        initial_workers=1,
        max_workers=1,
        queue_capacity_frames=2,
    )

    sim = Simulator(config)

    sim.step(
        "hold",
        [0],
    )

    assert sim.dropped_full > 0

    assert len(sim.queue) <= (
        config.queue_capacity_frames
    )

    assert sim.arrived == frame_accounting(
        sim
    )


def test_reported_capacity_matches_processing_time():
    config = SimulatorConfig(
        camera_fps=1.0,
        service_seconds=0.2,
        startup_seconds=0.5,
        deadline_seconds=5.0,
        decision_seconds=0.5,
        initial_workers=2,
        max_workers=2,
        queue_capacity_frames=20,
    )

    sim = Simulator(config)

    result = sim.step(
        "hold",
        [],
    )

    effective_service = (
        sim.service_ticks
        * config.tick_seconds
    )

    expected_capacity = (
        result["serving_workers"]
        / effective_service
    )

    assert result[
        "capacity_fps"
    ] == pytest.approx(
        expected_capacity
    )