import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "dqn_validation"


INCLUDE_PREFIXES = (
    "vanilla_seed",
    "nstep_seed",
    "dueling_seed",
    "dueling_nstep_seed",
)


def main():
    rows = []

    for path in sorted(RESULTS_DIR.glob("*/overall.json")):
        data = json.loads(path.read_text())

        model = Path(data.get("model", "")).stem

        if not model.startswith(INCLUDE_PREFIXES):
            continue

        dropped = (
            data.get("dropped_deadline", 0)
            + data.get("dropped_full", 0)
        )

        rows.append(
            {
                "model": model,
                "on_time_percentage": round(
                    data["on_time_percentage"],
                    2,
                ),
                "late": data["late"],
                "dropped": dropped,
                "worker_seconds": round(
                    data["worker_seconds"],
                    1,
                ),
                "average_workers": round(
                    data["average_workers"],
                    2,
                ),
                "p95_latency_seconds": round(
                    data["p95_latency_seconds"],
                    3,
                ),
                "scaling_changes": data["scaling_changes"],
                "blocked_actions": data["blocked_actions"],
            }
        )

    output = RESULTS_DIR / "dqn_comparison.csv"

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("DQN COMPARISON")
    print(
        f"{'MODEL':34}"
        f"{'ON-TIME':>10}"
        f"{'AVG W':>9}"
        f"{'P95':>9}"
        f"{'DROPPED':>11}"
        f"{'BLOCKED':>10}"
    )

    for row in rows:
        print(
            f"{row['model'][:34]:34}"
            f"{row['on_time_percentage']:>9.2f}%"
            f"{row['average_workers']:>9.2f}"
            f"{row['p95_latency_seconds']:>9.3f}"
            f"{row['dropped']:>11}"
            f"{row['blocked_actions']:>10}"
        )

    print()
    print(f"Saved comparison to {output}")


if __name__ == "__main__":
    main()