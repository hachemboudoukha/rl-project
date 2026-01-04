import numpy as np
from typing import Tuple, Optional, List
from .base_env import BaseEnvironment

class LineWorld(BaseEnvironment):
    """
    Line World Environment
    The agent moves on a line of positions.
    - Actions: 0 (left), 1 (right)
    - Reward: 1 at goal, 0 otherwise
    """
    
    def __init__(self, length: int = 7, start_pos: int = 3, goal_pos: Optional[int] = None):
        self.length = length
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos is not None else length - 1
        self.reset()
        
        self.actions = {0: "left", 1: "right"}
        self.n_actions = len(self.actions)
        self.n_states = length
        
    def reset(self) -> int:
        self.current_pos = self.start_pos
        self.done = False
        return self.current_pos
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self.done:
            return self.current_pos, 0.0, True, {}
        
        if action == 0:  # Left
            self.current_pos = max(0, self.current_pos - 1)
        elif action == 1:  # Right
            self.current_pos = min(self.length - 1, self.current_pos + 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        if self.current_pos == self.goal_pos:
            reward = 1.0
            self.done = True
        else:
            reward = 0.0
        
        info = {'position': self.current_pos}
        return self.current_pos, reward, self.done, info
    
    def get_actions(self, state: Optional[int] = None) -> List[int]:
        return list(self.actions.keys())
    
    def get_states(self) -> List[int]:
        return list(range(self.length))
    
    def is_terminal(self, state: int) -> bool:
        return state == self.goal_pos
    
    def render(self):
        line = ['_'] * self.length
        line[self.current_pos] = 'A'
        line[self.goal_pos] = 'G' if self.current_pos != self.goal_pos else 'X'
        print(''.join(['[' + cell + ']' for cell in line]))

    def simulate_step(self, state, action):
        if action == 0:
            next_state = max(0, state - 1)
        else:
            next_state = min(self.length - 1, state + 1)
        reward = 1.0 if next_state == self.goal_pos else 0.0
        done = next_state == self.goal_pos
        return next_state, reward, done
