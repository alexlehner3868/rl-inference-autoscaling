#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "Usage: bash Rayserve/run_ray_auto.sh results/rayserve_live/BATCH_DIRECTORY" >&2
  exit 2
fi

batch_dir="$1"
workloads=(workloads/validation/*.json)
if [[ ${#workloads[@]} -ne 20 ]]; then
  echo "Expected 20 validation workloads; found ${#workloads[@]}." >&2
  exit 1
fi

# Compare only with fresh results that include the current accounting fields.
python - "$batch_dir" <<'PY'
import json
import sys
from pathlib import Path

batch = Path(sys.argv[1])
workloads = sorted(path.stem for path in Path('workloads/validation').glob('*.json'))
for controller in ('fixed', 'queue', 'predictive'):
    for workload in workloads:
        summary = batch / f'{controller}_{workload}' / 'summary.json'
        if not summary.is_file():
            raise SystemExit(f'Missing external-controller run: {summary}')
        row = json.loads(summary.read_text())
        if not {'ready_worker_seconds', 'request_timeout_seconds', 'http_concurrency_limit'} <= row.keys():
            raise SystemExit(f'Run was made with an older runner; rerun it: {summary}')
PY

for workload in "${workloads[@]}"; do
  name="${workload##*/}"
  name="${name%.json}"
  output="$batch_dir/ray_auto_$name"
  log="$batch_dir/ray_auto_$name.log"
  if [[ -e "$output" || -e "$log" ]]; then
    echo "Output already exists for $name: $output or $log" >&2
    exit 1
  fi

  echo "Running Ray Serve autoscaling on $name"
  if ! python -m Rayserve.live_autoscale \
    --controller ray_auto \
    --max-workers 10 \
    --workload "$workload" \
    --duration 0 \
    --queue-size 50 \
    --deadline 1 \
    --request-timeout 4 \
    --output "$output" > "$log" 2>&1; then
    echo "Failed on $name. Last log lines:" >&2
    tail -40 "$log" >&2
    exit 1
  fi
done

echo "Ray Serve autoscaling results saved in: $batch_dir"
