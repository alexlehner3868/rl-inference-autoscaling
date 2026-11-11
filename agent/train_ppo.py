
import argparse
import csv
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from agent.train_dqn import make_env

root = Path(__file__).resolve().parent.parent
default_output = root / 'results/models/ppo_autoscaler'
validation_folder = root / 'results/ppo_validation'

# Train PPO
def train_ppo(timesteps=30000, seed=0, output=default_output):
    env = make_env('train')
    check_env(env, warn=True)

    model = PPO('MlpPolicy', env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
        verbose=1)

    model.learn(total_timesteps=timesteps)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    env.close()
    print('Model saved:', str(output) + '.zip')
    return model
# Test the trained PPO model on every validation workload
def validate_ppo(model):
    env = make_env('validation')
    workload_files = sorted((root / 'workloads/validation').glob('*.json'))
    all_results = []
    for i, file in enumerate(workload_files):
        obs, info = env.reset(options={'workload_index': i})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        sim = env.simulator
        on_time_percentage = 100 * sim.on_time / sim.arrived if sim.arrived else 0
        average_workers = sim.worker_seconds / env.workload['duration_seconds']
        result = {
            'workload': file.stem,
            'arrived': sim.arrived,
            'on_time': sim.on_time,
            'late': sim.late,
            'dropped_deadline': sim.dropped_deadline,
            'dropped_full': sim.dropped_full,
            'on_time_percentage': on_time_percentage,
            'worker_seconds': sim.worker_seconds,
            'average_workers': average_workers,
            'scaling_changes': sim.action_counts['add'] + sim.action_counts['remove'],
            'add_actions': sim.action_counts['add'],
            'remove_actions': sim.action_counts['remove']}
        all_results.append(result)
        print(file.stem, '- on time:', round(on_time_percentage, 2), '%', '- avg workers:', round(average_workers, 2),'- scaling changes:', result['scaling_changes'])
    env.close()

    # Calculate validation performance
    total_arrived = sum(row['arrived'] for row in all_results)
    total_on_time = sum(row['on_time'] for row in all_results)
    total_worker_seconds = sum(row['worker_seconds'] for row in all_results)
    total_scaling = sum(row['scaling_changes'] for row in all_results)
    total_duration = sum(300 for row in all_results)

    overall_on_time = 100 * total_on_time / total_arrived if total_arrived else 0
    average_workers = total_worker_seconds / total_duration
    average_scaling = total_scaling / len(all_results)

    print('\nValidation Summary')
    print('Overall on-time:', round(overall_on_time, 2), '%')
    print('Average workers:', round(average_workers, 2))
    print('Average scaling changes:', round(average_scaling, 2))
    print('Total frames:', total_arrived)
    print('On-time frames:', total_on_time)
    # Save the results
    validation_folder.mkdir(parents=True, exist_ok=True)
    with open(validation_folder / 'comparison.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print('Results saved in:', validation_folder / 'comparison.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=default_output)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    if args.eval_only:
        model = PPO.load(str(args.output))
    else:
        model = train_ppo(args.timesteps, args.seed, args.output)
    validate_ppo(model)
