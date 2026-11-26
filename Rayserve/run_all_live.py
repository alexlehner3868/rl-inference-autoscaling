
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

root = Path(__file__).resolve().parent.parent

# Settings
controllers = ['fixed', 'queue', 'dqn', 'ppo']
workload_folder = root / 'workloads/validation'
dqn_model = root / 'results/models/nstep_seed2_30k.zip'
ppo_model = root / 'results/models/ppo_autoscaler.zip'
output_folder = root / 'results/rayserve_live' / datetime.now().strftime('overnight_%Y%m%d_%H%M%S')
output_folder.mkdir(parents=True, exist_ok=True)
workload_files = sorted(workload_folder.glob('*.json'))

# Run all controllers on every validation workload
for file in workload_files:
    for controller in controllers:
        print('Running:', controller, file.stem, flush=True)
        folder = output_folder / (controller + '_' + file.stem)
        command = [sys.executable, '-m', 'Rayserve.live_autoscale',
            '--controller', controller,
            '--workload', str(file),
            '--duration', '0',
            '--output', str(folder)]
        if controller == 'fixed':
            command.extend(['--workers', '4'])

        elif controller == 'dqn':
            command.extend(['--model', str(dqn_model)])

        elif controller == 'ppo':
            command.extend(['--model', str(ppo_model)])

        # Save terminal output in a log file
        log_file = output_folder / (controller + '_' + file.stem + '.log')
        with open(log_file, 'w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            print('Completed:', controller, file.stem, flush=True)
        else:
            print('FAILED:', controller, file.stem, '- See:', log_file, flush=True)

        time.sleep(2)

print('Finished all experiments')
print('Results saved in:', output_folder)
