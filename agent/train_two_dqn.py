import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from torch.nn import functional as F

from agent.train_dqn import ROOT, create_dqn, make_env

"""
A is the DQN agent that we train to run the autoscaler 
    - selector is the agent 
    - selector.q_net is the NN 

During training, B helps A train by scoring the action that A pickts. 
"""

DEFAULT_MODEL_PATH = ROOT / "results" / "models" / "two_dqn_interleaved_a.zip"

# Subclass of DQN
class SelectingDQN(DQN):

    #We override this function becuase in the original, we occasionaly copy A's online weights in the the target slot but we dont want to do that becuase our target slot is holding B's weights.
    def _on_step(self) -> None:
        # Update A's exploration rate and record it 
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    # Define the overridden training loop
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Put A into training mode 
        self.policy.set_training_mode(True)
        # Set learning rate 
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            # Sample a batch from A's relay buffer for each step
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Get the discount (either from the batch or use gamma)
            discounts = getattr(batch, "discounts", None)
            if discounts is None:
                discounts = self.gamma

            with torch.no_grad():
                # Get the index of the best action from A
                next_actions = self.q_net(batch.next_observations).argmax(dim=1, keepdim=True)
                # Get the score from B for the same best action
                next_values = self.q_net_target(batch.next_observations).gather(1, next_actions)
                # Calculate the target that we use for loss calculations 
                targets = batch.rewards + (1 - batch.dones) * discounts * next_values

            # Get A's score for the action it took and calculate loss 
            predictions = self.q_net(batch.observations).gather(1, batch.actions.long())
            loss = F.smooth_l1_loss(predictions, targets)

            # Compute gradients 
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)

            # Update the policy
            self.policy.optimizer.step()
            losses.append(loss.item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def train_rollout(model, callback):
    # Get a batch of trajectory steps 
    model.collect_rollouts(model.get_env(), callback=callback, train_freq=model.train_freq, replay_buffer=model.replay_buffer, learning_starts=model.learning_starts, log_interval=4)

    # Perform training once we've collected at least 1000 steps 
    if model.num_timesteps > model.learning_starts:
        model.train(gradient_steps=model.gradient_steps, batch_size=model.batch_size)

def train_two_dqn(total_timesteps=30000, seed=0, save_path=DEFAULT_MODEL_PATH):
    # Split the data into two distinct sets (one for each DQN)
    if total_timesteps <= 2000 or total_timesteps % 8:
        raise ValueError("Total timesteps must be a multiple of 8 greater than 2000")
    steps_per_network = total_timesteps // 2

    with ExitStack() as cleanup:
        # Create two training enviornments, each drawing from same pool of data but maintainng their own states
        env_a = make_env("train")
        cleanup.callback(env_a.close)
        env_b = make_env("train")
        cleanup.callback(env_b.close)

        # Create selector with the custom SelectingDQN class
        selector = create_dqn(env_a, seed=seed, algorithm=SelectingDQN)
        # Create evaluator with the regular DQN class
        evaluator = create_dqn(env_b, seed=seed + 1)

        # Disable gradients on A's copy of B (we refresh in the loop)
        selector.q_net_target.requires_grad_(False) 

        # init each agent's training state
        _, callback_a = selector._setup_learn(steps_per_network, tb_log_name="network_a")
        _, callback_b = evaluator._setup_learn(steps_per_network, tb_log_name="network_b")
        callback_a.on_training_start(locals(), globals())
        callback_b.on_training_start(locals(), globals())
        print(f"Training A and B together for {steps_per_network:,} steps each", flush=True)

        # Do training but loop between selector and evaluator 
        while selector.num_timesteps < steps_per_network:
            # B collects is training samples and trains
            train_rollout(evaluator, callback_b)
            # Copy weights over from B to A's slot to store B's weights (note: not overwriting A's weights)
            selector.q_net_target.load_state_dict(evaluator.q_net.state_dict())
            # A collects samples and then trains
            train_rollout(selector, callback_a)

        # Training has finished 
        callback_a.on_training_end()
        callback_b.on_training_end()

        save_path = Path(save_path)
        
        if not save_path.suffix:
            save_path = save_path.with_suffix(".zip")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        selector.save(save_path)
       
        return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=30000, help="Total steps across A and B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    print(f"Saved network A checkpoint to {train_two_dqn(args.timesteps, args.seed, args.output)}")


if __name__ == "__main__":
    main()
