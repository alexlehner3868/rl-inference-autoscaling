bash <<'BASH'
set -e
shopt -s nullglob

workloads=(workloads/validation/*.json)
if [[ ${#workloads[@]} -ne 20 ]]; then
  echo "Expected 20 validation workloads; found ${#workloads[@]}."
  exit 1
fi

run_dir="$(mktemp -d results/rayserve_live/all20_XXXXXX)"
echo "Saving results in: $run_dir"

run_one() {
  local controller="$1"
  local name="$2"
  local workload="$3"
  shift 3

  local log="$run_dir/${controller}_${name}.log"
  echo "Running $controller on $name"

  if ! python -m Rayserve.live_autoscale \
    --controller "$controller" \
    --max-workers 10 \
    --workload "$workload" \
    --duration 0 \
    --queue-size 50 \
    --deadline 1 \
    --output "$run_dir/${controller}_${name}" \
    "$@" > "$log" 2>&1; then
    echo "Failed: $controller on $name. Last log lines:"
    tail -40 "$log"
    exit 1
  fi
}

for workload in "${workloads[@]}"; do
  name="${workload##*/}"
  name="${name%.json}"

  run_one fixed "$name" "$workload" \
    --workers 9

  run_one queue "$name" "$workload" \
    --add-threshold 3 \
    --remove-threshold 0 \
    --patience 20

  run_one predictive "$name" "$workload" \
    --forecast-window 5 \
    --predictive-down-steps 5 \
    --calibrated-service 0.07197132544888889 \
    --calibrated-startup 2.3057613082
done

python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
print("\nController   Runs   On-time %   Worker-seconds   Worst workload %")

for controller in ("fixed", "queue", "predictive"):
    files = sorted(run_dir.glob(f"{controller}_*/summary.json"))
    rows = [json.loads(path.read_text()) for path in files]
    arrived = sum(row["arrived"] for row in rows)
    on_time = sum(row["on_time"] for row in rows)
    worker_seconds = sum(row["worker_seconds"] for row in rows)
    worst = min(row["on_time_percentage"] for row in rows)

    print(
        f"{controller:11} {len(rows):4} "
        f"{100 * on_time / arrived:11.2f} "
        f"{worker_seconds:16.0f} "
        f"{worst:18.2f}"
    )

print(f"\nIndividual summaries and logs: {run_dir}")
PY
BASH