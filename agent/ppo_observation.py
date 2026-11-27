import numpy as np
from agent.dqn_env import InferenceScalingEnv
from gymnasium import spaces

PPO_SCALE = np.array([50.0, 1.0, 120.0, 10.0, 10.0], dtype=np.float32)

def normalize_ppo_observation(observation):
    return np.asarray(observation, dtype=np.float32) / PPO_SCALE

class NormalizedPPOEnv(InferenceScalingEnv):
    def _get_observation(self):
        return normalize_ppo_observation(super()._get_observation())



class TargetPPOEnv(NormalizedPPOEnv):
    ACTIONS = {i: ("target", i + 1) for i in range(10)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.max_workers != 10:
            raise ValueError("TargetPPOEnv requires max_workers=10")
        self.action_space = spaces.Discrete(10)