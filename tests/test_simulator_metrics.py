import csv
import json
from pathlib import Path
import tempfile
import unittest

from Simulator.Simulator_queue import Simulator, SimulatorConfig


class SimulatorMetricsTests(unittest.TestCase):
    def assert_frame_accounting(self, sim):
        processing = sum(w["frame"] is not None for w in sim.workers)
        self.assertEqual(sim.arrived, sim.on_time + sim.late + sim.dropped_deadline
                         + sim.dropped_full + len(sim.queue) + processing)
        for name in ("arrived", "on_time", "late", "dropped_deadline", "dropped_full"):
            self.assertEqual(getattr(sim, name), sum(c[name] for c in sim.camera_stats.values()))

    def test_startup_capacity_changes_inside_decision_interval(self):
        sim = Simulator(SimulatorConfig(camera_fps=20, service_seconds=0.1,
                                       startup_seconds=0.35, deadline_seconds=10))
        row = sim.step("add", [0])
        self.assertEqual(row["demand_fps"], 20)
        self.assertEqual(row["capacity_fps"], 20)
        self.assertEqual(row["serving_workers"], 2)
        self.assertEqual(row["starting_workers"], 0)
        self.assertAlmostEqual(row["worker_seconds"], 2)
        self.assertEqual(len(sim.capacity_timeline), 2)
        self.assertEqual(sim.capacity_timeline[0], (0, 10))
        self.assertAlmostEqual(sim.capacity_timeline[1][0], 0.35)
        self.assertEqual(sim.capacity_timeline[1][1], 20)
        self.assert_frame_accounting(sim)

    def test_retiring_worker_finishes_but_provides_no_new_capacity(self):
        sim = Simulator(SimulatorConfig(service_seconds=0.1, decision_seconds=0.05,
                                       initial_workers=2))
        sim.step("hold", [0, 1])
        row = sim.step("remove", [0, 1])
        self.assertEqual(row["workers"], 2)
        self.assertEqual(row["serving_workers"], 1)
        self.assertEqual(row["retiring_workers"], 1)
        self.assertEqual(row["capacity_fps"], 10)
        self.assertAlmostEqual(row["worker_seconds"], 0.2)
        self.assertAlmostEqual(sim.capacity_timeline[-1][0], 0.05)
        sim.step("hold", [0, 1])
        self.assertEqual(len(sim.workers), 1)
        self.assert_frame_accounting(sim)

    def test_drops_lateness_and_per_camera_accounting(self):
        sim = Simulator(SimulatorConfig(camera_fps=20, service_seconds=0.2,
                                       deadline_seconds=0.1, decision_seconds=0.3,
                                       queue_capacity_frames=1))
        sim.step("remove", [0, 1])  # Blocked: already at the minimum.
        sim.step("hold", [0, 1])
        self.assertGreater(sim.late, 0)
        self.assertGreater(sim.dropped_deadline, 0)
        self.assertGreater(sim.dropped_full, 0)
        self.assertEqual(sim.action_counts["blocked"], 1)
        self.assertTrue(all(latency > 0.1 for latency in sim.latencies_seconds))
        self.assert_frame_accounting(sim)

    def test_exports_and_unfinished_frame_reporting(self):
        sim = Simulator(SimulatorConfig(decision_seconds=0.001))
        sim.step("hold", [0])
        with tempfile.TemporaryDirectory() as temporary:
            output = sim.save_results(Path(temporary) / "run", controller="fixed", workload="short")
            self.assertEqual({p.name for p in output.iterdir()},
                             {"summary.json", "history.csv", "demand_vs_capacity.png"})
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["controller"], "fixed")
            self.assertEqual(summary["unfinished_frames"], 1)
            self.assertEqual(summary["on_time_percentage"], 0)
            self.assertIsNone(summary["completed_latency_seconds"]["p95"])
            self.assertEqual(summary["per_camera"]["0"]["processing"], 1)
            with (output / "history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["interval_start_seconds"]), 0)
            self.assertEqual((output / "demand_vs_capacity.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_zero_arrivals_are_reported_without_invalid_json(self):
        sim = Simulator(SimulatorConfig(decision_seconds=0.001))
        with self.assertRaises(ValueError):
            sim.save_results("unused")
        sim.step("hold", [])
        with tempfile.TemporaryDirectory() as temporary:
            output = sim.save_results(Path(temporary) / "empty_demand")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["arrived"], 0)
            self.assertIsNone(summary["on_time_percentage"])
            self.assertEqual(summary["per_camera"], {})


if __name__ == "__main__":
    unittest.main()
