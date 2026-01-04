import numpy as np
from typing import Tuple, List, Optional
from .base_env import BaseEnvironment
import sys
import os

# Add root to sys.path to import secret_envs_wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from rl.environments.secret import secret_envs_wrapper

class SecretEnvWrapper(BaseEnvironment):
    """
    Generic wrapper for secret environments (0, 1, 2, 3)
    to make them compatible with common BaseEnvironment interface.
    """
    def __init__(self, env_id: int):
        self.env_id = env_id
        if env_id == 0:
            self.env = secret_envs_wrapper.SecretEnv0()
        elif env_id == 1:
            self.env = secret_envs_wrapper.SecretEnv1()
        elif env_id == 2:
            self.env = secret_envs_wrapper.SecretEnv2()
        elif env_id == 3:
            self.env = secret_envs_wrapper.SecretEnv3()
        else:
            raise ValueError(f"Invalid Secret Env ID: {env_id}")
        
        self.n_actions = self.env.num_actions()
        self.n_states = self.env.num_states()
        self.reset()

    def reset(self) -> int:
        self.env.reset()
        return self.env.state_id()

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        prev_score = self.env.score()
        self.env.step(action)
        new_state = self.env.state_id()
        reward = self.env.score() - prev_score
        done = self.env.is_game_over()
        return new_state, reward, done, {}

    def get_actions(self, state: Optional[int] = None) -> List[int]:
        # Note: In secret environments, available actions might depend on state
        # but here we return all possible action IDs for the environment.
        return list(range(self.n_actions))

    def get_states(self) -> List[int]:
        return list(range(self.n_states))

    def is_terminal(self, state: int) -> bool:
        # This is tricky because the wrapper only tells us if the CURRENT state is game over
        # For DP, we might need a better way. 
        # But for TD/MC, this should be fine if we use the step result.
        return self.env.is_game_over()

    def render(self):
        self.env.display()

    @property
    def state(self):
        return self.env.state_id()

    # Note: Secret environments don't seem to support setting state directly easily 
    # except via from_random_state which creates a new instance.
